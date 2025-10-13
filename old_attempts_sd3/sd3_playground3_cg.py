#!/usr/bin/env python3
# sd3_probe_ooo_streamed.py
#
# Probe Stable Diffusion 3 transformer blocks for zero-shot odd-one-out (OOO) accuracy
# at a single specified noise level, using streaming mini-batches (OOM-safe).
# Requires: diffusers, torch, torchvision, tqdm, numpy, and your local `things` + `helpers`.

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage
from tqdm import tqdm

# Local modules expected to be available in your env
from things import THINGSBehavior
import helpers


# ----------------------------
# Utilities
# ----------------------------

def seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def spatial_mean_pool(x: torch.Tensor) -> torch.Tensor:
    """Repeatedly mean-pool trailing dims until tensor is [B, D]."""
    while x.ndim > 2:
        x = x.mean(dim=-1)
    return x

def get_scheduler_timestep_from_percent(
    scheduler,
    percent: float,
    num_inference_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Map noise percentage (0..1) to a scheduler timestep on the current grid."""
    percent = float(np.clip(percent, 0.0, 1.0))
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps  # length = num_inference_steps
    if timesteps.ndim != 1 or len(timesteps) == 0:
        raise RuntimeError("Scheduler did not provide a valid timesteps tensor.")
    # 0% => last (low noise), 100% => first (high noise)
    idx = int(round((len(timesteps) - 1) * (1.0 - percent)))
    idx = max(0, min(idx, len(timesteps) - 1))
    t = timesteps[idx]
    if not torch.is_tensor(t):
        t = torch.tensor(t, device=device, dtype=torch.long)
    else:
        t = t.to(device=device)
        if t.dtype not in (torch.long, torch.int64):
            t = t.long()
    return t

def add_noise_with_scheduler_fallback(
    latents_z0: torch.Tensor,
    noise: torch.Tensor,
    timestep: torch.Tensor,
    scheduler,
) -> torch.Tensor:
    """
    Produce z_t consistent with the scheduler if possible. Try:
    1) scheduler.add_noise
    2) DDPM-style alphas_cumprod
    3) k-diffusion-like sigmas
    4) last resort: scaled additive noise
    """
    # 1) Direct support
    if hasattr(scheduler, "add_noise") and callable(getattr(scheduler, "add_noise")):
        return scheduler.add_noise(latents_z0, noise, timestep)

    # 2) DDPM path using alphas_cumprod
    try:
        t = timestep[0] if timestep.ndim > 0 else timestep
        t = int(t.item())
        if hasattr(scheduler, "alphas_cumprod"):
            alphas_cumprod = scheduler.alphas_cumprod
            t = max(0, min(t, len(alphas_cumprod) - 1))
            alpha_bar = alphas_cumprod[t].to(device=latents_z0.device, dtype=latents_z0.dtype)
            alpha = torch.sqrt(alpha_bar)
            sigma = torch.sqrt(1.0 - alpha_bar)
            return alpha * latents_z0 + sigma * noise
    except Exception:
        pass

    # 3) Sigmas path
    try:
        if hasattr(scheduler, "sigmas") and scheduler.sigmas is not None:
            sigmas = scheduler.sigmas.to(device=latents_z0.device)
            step_val = float((timestep[0] if timestep.ndim > 0 else timestep).item())
            if hasattr(scheduler, "timesteps"):
                ts = scheduler.timesteps
                diffs = torch.abs(ts.float() - step_val)
                idx = int(torch.argmin(diffs).item())
            else:
                idx = min(len(sigmas) - 1, max(0, len(sigmas) // 2))
            sigma = sigmas[idx].to(dtype=latents_z0.dtype)
            return latents_z0 + sigma * noise
    except Exception:
        pass

    # 4) Fallback
    return latents_z0 + 0.5 * noise

def parse_block_spec(spec: Optional[str], count: int) -> List[int]:
    """
    Parse a block selection string into concrete indices.
      "all" -> [0..count-1]
      "last" / "highest" -> [count-1]
      "0,3,5-8,-1" -> [0,3,5,6,7,8,count-1]
      "-3--1" -> [count-3, count-2, count-1]
    """
    if spec is None or spec.strip().lower() in ("all", ""):
        return list(range(count))
    s = spec.strip().lower().replace(" ", "")
    if s in ("last", "highest", "top", "max"):
        return [count - 1]

    out: List[int] = []

    def add_idx(v: int):
        if v < 0:
            v = count + v
        v = max(0, min(count - 1, v))
        if v not in out:
            out.append(v)

    for token in s.split(","):
        if not token:
            continue
        if token == "all":
            for i in range(count):
                add_idx(i)
            continue
        if "-" in token:
            # handle ranges (including negatives), e.g., "5-8" or "-3--1"
            try:
                a_str, b_str = token.split("-", 1)
                if b_str == "" and token.count("-") >= 2:
                    a_str, b_str = token.rsplit("-", 1)
                a = int(a_str)
                b = int(b_str)
                if a < 0: a = count + a
                if b < 0: b = count + b
                a = max(0, min(count - 1, a))
                b = max(0, min(count - 1, b))
                rng = range(a, b + 1) if a <= b else range(a, b - 1, -1)
                for v in rng:
                    add_idx(v)
            except Exception:
                try:
                    add_idx(int(token))
                except Exception:
                    pass
        else:
            try:
                add_idx(int(token))
            except Exception:
                pass

    if not out:
        raise ValueError(f"No valid blocks parsed from spec '{spec}'.")
    out.sort()
    return out


# ----------------------------
# Prober
# ----------------------------

class SD3TransformerProber:
    """
    Probes SD3 transformer blocks at a chosen timestep (derived from a noise %),
    capturing each block's activations as features.
    """
    def __init__(
        self,
        checkpoint: str = "stabilityai/stable-diffusion-3-medium-diffusers",
        device: str = "cuda",
        fp16_on_cuda: bool = True,
        seed: int = 0,
        enable_memory_savers: bool = True,
    ):
        super().__init__()
        self.device_str = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.dtype = torch.float16 if (self.device_str == "cuda" and fp16_on_cuda) else torch.float32
        self.device = torch.device(self.device_str)
        seed_all(seed)

        from diffusers import StableDiffusion3Pipeline
        from diffusers.utils import logging as diffusers_logging
        diffusers_logging.disable_progress_bar()

        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            checkpoint,
            torch_dtype=self.dtype,
        ).to(self.device)

        # Components
        self.vae = self.pipe.vae
        self.transformer = self.pipe.transformer
        self.scheduler = self.pipe.scheduler

        if self.transformer is None:
            raise ValueError("SD3 transformer component not found in pipeline.")

        blocks = getattr(self.transformer, "transformer_blocks", None)
        if blocks is None or len(blocks) == 0:
            raise ValueError("No transformer_blocks found on SD3 transformer.")
        self.blocks = list(blocks)

        # Config-based dims
        self.joint_dim = self.transformer.config.joint_attention_dim
        self.pooled_dim = self.transformer.config.pooled_projection_dim

        # VAE scaling factor
        self.scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)

        # RNG for noise
        self.generator = torch.Generator(device=self.device)

        # Quiet progress bars
        try:
            self.pipe.set_progress_bar_config(disable=True, leave=False)
        except Exception:
            pass

        # Memory savers
        if enable_memory_savers:
            try:
                if hasattr(self.vae, "enable_tiling"):
                    self.vae.enable_tiling()
                if hasattr(self.vae, "enable_slicing"):
                    self.vae.enable_slicing()
                if hasattr(self.pipe, "enable_attention_slicing"):
                    self.pipe.enable_attention_slicing(1)
                if hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
                    self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

    @torch.no_grad()
    def encode_images_to_latents(self, imgs_01: torch.Tensor) -> torch.Tensor:
        """
        imgs_01: [B, 3, H, W] in [0,1]
        returns z0: [B, C, H', W'] in VAE latent space (deterministic mode), scaled.
        """
        x = imgs_01.to(device=next(self.vae.parameters()).device, dtype=self.vae.dtype)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = x * 2.0 - 1.0  # [0,1] -> [-1,1]
        posterior = self.vae.encode(x).latent_dist
        z = posterior.mode()  # deterministic
        z = z * self.scaling_factor
        return z

    @torch.no_grad()
    def make_uncond_text_embeds(self, batch_size: int, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        prompt_embeds = torch.zeros(
            (batch_size, 77, self.joint_dim),
            device=self.device,
            dtype=dtype,
        )
        pooled = torch.zeros(
            (batch_size, self.pooled_dim),
            device=self.device,
            dtype=dtype,
        )
        return prompt_embeds, pooled

    @torch.no_grad()
    def forward_one_block(
        self,
        latents_zt: torch.Tensor,
        timestep: torch.Tensor,
        block_idx: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Run transformer once and capture features from the chosen block; returns [B, D].
        Ensures timestep is a 1-D tensor of length B (fixes SD3 time embedding requirement).
        """
        B = latents_zt.shape[0]

        # ---- Ensure timestep is 1-D of length B (float32) ----
        if not torch.is_tensor(timestep):
            timestep = torch.tensor(timestep, device=self.device)
        timestep = timestep.to(self.device)
        if timestep.ndim == 0:
            timestep = timestep.repeat(B)
        elif timestep.ndim == 1 and timestep.shape[0] == 1:
            timestep = timestep.repeat(B)
        else:
            if timestep.shape[0] != B:
                timestep = timestep[:1].repeat(B)
        if timestep.dtype not in (torch.float32, torch.float64):
            timestep = timestep.float()
        # -------------------------------------------------------

        prompt_embeds, pooled = self.make_uncond_text_embeds(B, dtype=dtype)

        activations: List[torch.Tensor] = []

        def hook_fn(_m, _inp, out):
            sel = None
            if isinstance(out, torch.Tensor):
                sel = out
            elif isinstance(out, (list, tuple)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                sel = out[0]
            elif isinstance(out, dict) and "hidden_states" in out and isinstance(out["hidden_states"], torch.Tensor):
                sel = out["hidden_states"]
            if sel is not None:
                activations.append(sel.detach())

        handle = self.blocks[block_idx].register_forward_hook(hook_fn)
        try:
            _ = self.transformer(
                hidden_states=latents_zt,
                timestep=timestep,  # now [B]
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled,
                return_dict=True,
            )
        finally:
            handle.remove()

        if len(activations) == 0:
            raise RuntimeError(f"No activations captured from transformer_blocks[{block_idx}].")

        feats = spatial_mean_pool(activations[0].float())  # [B, D]
        return feats


# ----------------------------
# OOO evaluation
# ----------------------------

def compute_ooo_acc(features: np.ndarray, triplets: np.ndarray) -> float:
    """Zero-shot OOO accuracy using cosine distance and temperature=1.0."""
    choices, *_ = helpers.get_predictions(features, triplets, temperature=1.0, dist="cosine")
    return helpers.accuracy(choices)


# ----------------------------
# Main pipeline
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        "Probe SD3 transformer blocks for zero-shot OOO accuracy at a single noise level (streamed, OOM-safe)."
    )
    parser.add_argument("--things_root", type=str, default="/home/space/thesis/things_starterpack/things_data")
    parser.add_argument("--checkpoint", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cc0", action="store_true", help="Use CC0 subset if available in your THINGSBehavior impl")

    # Data / extraction
    parser.add_argument("--num_extractions", type=int, default=None, help="Limit number of images processed")
    parser.add_argument("--batch_size", type=int, default=1, help="Mini-batch size for VAE+transformer")
    parser.add_argument("--save_k", type=int, default=5, help="Save first-K originals and features for inspection")
    parser.add_argument("--save_dir", type=str, default="probe_outputs")

    # Noise / scheduler
    parser.add_argument("--noise_percent", type=float, default=0.50, help="Noise level as 0..1 (e.g., 0.50 for 50%)")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Scheduler timesteps grid size")

    # Block selection
    parser.add_argument(
        "--blocks",
        type=str,
        default="all",
        help=("Which transformer blocks to probe. Examples: "
              "'all', 'last', '0,3,5-8,-1', '-3--1'. Negative indices follow Python semantics.")
    )

    # Memory toggles
    parser.add_argument("--disable_memory_savers", action="store_true",
                        help="Disable VAE tiling/slicing and attention slicing/xFormers")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    stages = [
        "Initialize model/prober",
        "Parse blocks",
        "Load dataset",
        "Compute fixed timestep",
        "Streamed extraction",
        "Compute OOO accuracy & save",
    ]
    pbar = tqdm(total=len(stages), desc="Pipeline", position=0, leave=True,
                bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt}")

    # 1) Init model/prober
    prober = SD3TransformerProber(
        checkpoint=args.checkpoint,
        device=args.device,
        fp16_on_cuda=True,
        seed=args.seed,
        enable_memory_savers=not args.disable_memory_savers,
    )
    pbar.update(1)

    # 2) Parse blocks
    try:
        selected_blocks = parse_block_spec(args.blocks, count=len(prober.blocks))
    except Exception as e:
        raise SystemExit(f"Failed to parse --blocks '{args.blocks}': {e}")
    tqdm.write(f"Will probe transformer blocks: {selected_blocks}")
    pbar.update(1)

    # 3) Load dataset (we'll iterate it streamed)
    transform = Compose([Resize(512), ToTensor()])
    dataset = THINGSBehavior(root=args.things_root, aligned=False, download=False, transform=transform, cc0=args.cc0)
    total_N = len(dataset)
    if args.num_extractions is not None and args.num_extractions > 0:
        total_N = min(total_N, args.num_extractions)
    pbar.update(1)

    # 4) Compute fixed timestep (single noise level for entire run)
    t = get_scheduler_timestep_from_percent(
        prober.scheduler, args.noise_percent, args.num_inference_steps, prober.device
    )
    pbar.update(1)

    # 5) Streamed extraction over the dataset
    per_block_feats: Dict[int, List[torch.Tensor]] = {blk: [] for blk in selected_blocks}
    seen = 0
    saved_imgs = 0
    prober.generator.manual_seed(args.seed)

    loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=False
    )
    batches_needed = int(np.ceil(total_N / max(1, args.batch_size)))

    with torch.no_grad():
        for _, batch in enumerate(tqdm(loader, total=batches_needed,
                                       desc="Extracting features (streamed)", unit="batch")):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]

            # Respect num_extractions
            if args.num_extractions is not None and args.num_extractions > 0:
                remaining = args.num_extractions - seen
                if remaining <= 0:
                    break
                if batch.shape[0] > remaining:
                    batch = batch[:remaining]

            # Save first-K originals
            if args.save_k and saved_imgs < args.save_k:
                to_save = min(batch.shape[0], args.save_k - saved_imgs)
                for bi in range(to_save):
                    pil_img = ToPILImage()(batch[bi].detach().cpu())
                    pil_img.save(os.path.join(args.save_dir, f"orig_{saved_imgs+1:03d}.png"))
                    saved_imgs += 1

            # 1) Encode batch -> z0 (deterministic)
            z0 = prober.encode_images_to_latents(batch)  # [B, Cz, Hz, Wz]

            # 2) Make z_t for this batch (scheduler-consistent forward noising)
            # Use torch.randn(..., generator=...) for broad PyTorch compatibility
            noise = torch.randn(z0.shape, dtype=z0.dtype, device=z0.device, generator=prober.generator)
            zt = add_noise_with_scheduler_fallback(z0, noise, t, prober.scheduler)

            # 3) Run requested blocks, append features
            for blk in selected_blocks:
                feats = prober.forward_one_block(
                    latents_zt=zt, timestep=t, block_idx=blk, dtype=z0.dtype
                )  # [B, D]
                per_block_feats[blk].append(feats.detach().cpu())

            seen += batch.shape[0]

            # Keep VRAM tidy between batches
            if prober.device.type == "cuda":
                torch.cuda.empty_cache()

    # Concatenate per-block tensors -> numpy arrays
    feats_by_block_np: Dict[int, np.ndarray] = {
        blk: torch.cat(per_block_feats[blk], dim=0).numpy().astype(np.float32)
        for blk in selected_blocks
    }
    pbar.update(1)

    # 6) Load triplets and compute per-block OOO accuracy
    triplets = dataset.get_triplets()
    triplets = np.asarray(triplets)
    if triplets.ndim >= 2:
        N = seen
        mask = (triplets < N).all(axis=1)
        triplets = triplets[mask]
    else:
        tqdm.write("Triplets format unexpected; using as-is.")

    # Save first-K feature rows (optional)
    if args.save_k and args.save_k > 0:
        try:
            k = min(args.save_k, next(iter(feats_by_block_np.values())).shape[0])
            for blk in selected_blocks:
                arr = feats_by_block_np[blk]
                out_path = os.path.join(args.save_dir, f"features_block{blk}_first{k}.txt")
                np.savetxt(out_path, arr[:k], fmt="%.6f")
        except Exception as e:
            tqdm.write(f"Failed to save features preview: {e}")

    results: List[Tuple[int, float]] = []
    for blk in selected_blocks:
        feats = feats_by_block_np[blk]
        try:
            acc = compute_ooo_acc(feats, triplets) if triplets.size > 0 else float("nan")
        except Exception as e:
            tqdm.write(f"Block {blk}: failed to compute accuracy ({e}); setting NaN.")
            acc = float("nan")
        results.append((blk, acc))

    results.sort(key=lambda x: x[0])
    print("\nZero-shot OOO accuracy per transformer block "
          f"(noise={args.noise_percent:.2f}, steps={args.num_inference_steps}):")
    for blk, acc in results:
        print(f"  block {blk:02d}: {acc:.4f}")

    # Write table
    try:
        out_dir = os.path.join(args.save_dir, "sd3_results")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"ooo_acc_noise{args.noise_percent:.2f}_steps{args.num_inference_steps}.txt"), "w") as f:
            for blk, acc in results:
                f.write(f"{blk}\t{acc:.6f}\n")
        tqdm.write(f"Wrote per-block accuracy table to '{out_dir}'")
    except Exception as e:
        tqdm.write(f"Failed to write results file: {e}")

    pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    main()
