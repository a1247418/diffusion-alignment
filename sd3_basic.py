"""
SD3 feature extraction for THINGS OOO accuracy — fixed & paper-faithful.

What this does:
- Builds the THINGS dataset ONCE with aligned indices and passes that same instance everywhere.
- Extracts a representation from a chosen SD3 block at a FIXED noise level t using image2image:
    * uses num_inference_steps=--steps and strength=--noise_pct
    * stops on the FIRST hook call (corresponding to the chosen noise level)
    * robustly selects the block’s hidden states (or the largest tensor) from possibly nested outputs
- Pools correctly:
    * CNN tensors: mean over H,W  → (B, C)
    * Transformer/DiT tensors: mean over the TOKEN axis (the larger of dims 1 and 2) → (B, D)
- Adds a tiny dither to break cosine ties, then L2-normalizes features before OOO evaluation.
- Writes qualitative previews for the first K items and saves accuracy to sd3_results/.

CLI example:
python sd3_things_ooo.py --things_root /path/to/things --checkpoint stabilityai/stable-diffusion-3-medium-diffusers \
    --block_idx 8 --noise_pct 0.20 --steps 100 --num_extractions 500 --save_k 8
"""

import argparse
from typing import Optional
import os
import numpy as np
import torch
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage
from tqdm import tqdm
from things import THINGSBehavior
import helpers
from huggingface_hub import snapshot_download


# ------------------------
# Utilities
# ------------------------

class _EarlyStop(Exception):
    pass


def _select_tensor_from_output(out):
    """
    Prefer hidden states. Fall back to the largest tensor by numel.
    Handles SD3 MMDiT blocks that may return tensors, tuples/lists, or dicts.
    """
    candidates = []

    def consider(t):
        if isinstance(t, torch.Tensor) and t.ndim >= 2 and t.numel() > 0:
            candidates.append(t)

    if isinstance(out, torch.Tensor):
        consider(out)

    elif isinstance(out, dict):
        # Common keys holding features
        for k in ("hidden_states", "last_hidden_state", "sample", "x"):
            v = out.get(k, None)
            if isinstance(v, torch.Tensor):
                consider(v)
        # Scan the rest shallowly
        for v in out.values():
            if isinstance(v, torch.Tensor):
                consider(v)
            elif isinstance(v, (list, tuple)):
                for u in v:
                    if isinstance(u, torch.Tensor):
                        consider(u)

    elif isinstance(out, (list, tuple)):
        for v in out:
            if isinstance(v, torch.Tensor):
                consider(v)
            elif isinstance(v, (list, tuple)):
                for u in v:
                    if isinstance(u, torch.Tensor):
                        consider(u)

    if not candidates:
        return None
    return max(candidates, key=lambda t: t.numel())


def global_avg_pool(x: torch.Tensor) -> torch.Tensor:
    """
    Robust pooling:
      - CNN: (B, C, H, W) -> mean over H,W -> (B, C)
      - Transformer/DiT: (B, L, D) or (B, D, L) -> mean over TOKEN axis
        (choose the *larger* of dims 1 and 2 as tokens).
    """
    if x.ndim == 4:
        return x.mean(dim=(2, 3))
    if x.ndim == 3:
        token_axis = 1 if x.shape[1] >= x.shape[2] else 2
        return x.mean(dim=token_axis)
    if x.ndim == 2:
        return x
    # Fallback: flatten everything but batch
    return x.view(x.size(0), -1)


# ------------------------
# Encoder
# ------------------------

class SD3HighestBlockEncoder(torch.nn.Module):
    def __init__(
        self,
        checkpoint: str = "stabilityai/stable-diffusion-3-medium-diffusers",
        device: str = "cuda",
        block_idx: Optional[int] = 8,  # intermediate by default (often strongest)
    ):
        super().__init__()
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.checkpoint = checkpoint
        self.block_idx = block_idx

        # Lazy imports
        from diffusers import AutoPipelineForImage2Image
        from diffusers.utils import logging as diffusers_logging
        diffusers_logging.disable_progress_bar()

        # Build pipeline, match dtype to device
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            checkpoint,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

        # Also disable per-call bars just in case
        try:
            self.pipe.set_progress_bar_config(disable=True, leave=False)
        except Exception:
            pass

        # Prefer transformer (SD3 uses MMDiT)
        unet = getattr(self.pipe, "unet", None)
        if unet is None:
            transformer = getattr(self.pipe, "transformer", None)
            if transformer is None:
                raise ValueError("Could not find UNet/transformer in the SD3 pipeline.")

            sel_name = "(last unnamed)"
            block_module = None
            try:
                blocks = getattr(transformer, "transformer_blocks", None)
                if blocks is not None and len(blocks) > 0:
                    idx = len(blocks) - 1
                    if isinstance(self.block_idx, int):
                        idx = max(0, min(self.block_idx, len(blocks) - 1))
                    block_module = blocks[idx]
                    sel_name = f"transformer_blocks.{idx}"
            except Exception:
                block_module = None

            self.extraction_module = block_module if block_module is not None else list(transformer.modules())[-1]
            tqdm.write(
                f"SD3HighestBlockEncoder: using module 'transformer.{sel_name}' ({type(self.extraction_module).__name__})"
            )
        else:
            # UNet-style fallback (not typical for SD3)
            named = dict(unet.named_modules())
            candidate_names = []
            for i in reversed(range(0, 8)):
                candidate_names.append(f"up_blocks.{i}.resnets.1")
                candidate_names.append(f"up_blocks.{i}.attentions.1.transformer_blocks.0")
            candidate_names.append("mid_block")
            for i in reversed(range(0, 8)):
                candidate_names.append(f"down_blocks.{i}.resnets.1")
                candidate_names.append(f"down_blocks.{i}.attentions.1.transformer_blocks.0")
            found = None
            found_name = None
            for name in candidate_names:
                if name in named:
                    found = named[name]
                    found_name = name
                    break
            if found is None:
                found = list(named.values())[-1]
                try:
                    found_name = list(named.keys())[-1]
                except Exception:
                    found_name = "(last unnamed)"
            self.extraction_module = found
            tqdm.write(
                f"SD3HighestBlockEncoder: using module 'unet.{found_name}' ({type(self.extraction_module).__name__})"
            )

        self.generator = torch.Generator(device=self.device)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        *,
        noise_pct: float = 0.20,         # 20% noise
        num_inference_steps: int = 100,  # dense grid so strength≈t%
        seed: int = 42,
    ) -> torch.Tensor:
        """
        Single-step-at-t extraction via image2image:
        - num_inference_steps=--steps and strength=--noise_pct
        - stop on the FIRST hook call (corresponding to the chosen noise level)
        """
        # Safety: keep bars disabled
        try:
            self.pipe.set_progress_bar_config(disable=True, leave=False)
        except Exception:
            pass

        self.generator.manual_seed(seed)
        captured = []

        def hook_fn(_m, _inp, out):
            sel = _select_tensor_from_output(out)
            if sel is not None:
                captured.append(sel.detach())
                raise _EarlyStop  # first call = chosen t

        handle = self.extraction_module.register_forward_hook(hook_fn)

        try:
            try:
                self.pipe(
                    image=x.to(self.device),
                    strength=float(max(0.0, min(1.0, noise_pct))),
                    prompt="",                 # unconditional
                    guidance_scale=0.0,        # unconditional zero-shot
                    num_inference_steps=num_inference_steps,
                    generator=self.generator,
                )
            except _EarlyStop:
                pass
        finally:
            handle.remove()

        if not captured:
            raise RuntimeError("No activation captured at the chosen noise level.")
        return captured[0]


# ------------------------
# Feature extraction
# ------------------------

@torch.no_grad()
def extract_features(
    encoder: SD3HighestBlockEncoder,
    dataset: THINGSBehavior,
    *,
    pool: bool = True,
    num_extractions: Optional[int] = None,
    noise_pct: float = 0.20,
    steps: int = 100,
    save_k: int = 5,
    save_dir: str = "denoised",
) -> np.ndarray:
    """
    Extract features for the provided (aligned) THINGS dataset using a single-step
    activation at noise level t. Returns L2-normalized features as float32 numpy array.
    """
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    effective_total = len(dataset) if (not num_extractions or num_extractions <= 0) else min(len(dataset), num_extractions)

    if save_k and save_k > 0:
        os.makedirs(save_dir, exist_ok=True)

    feats = []
    with torch.no_grad():
        for idx, sample in enumerate(tqdm(loader, total=effective_total, desc="Extracting features@t", unit="img")):
            if num_extractions and num_extractions > 0 and idx >= num_extractions:
                break

            img = sample[0] if isinstance(sample, (list, tuple)) else sample  # (C,H,W) in [0,1]

            # Save originals for first K (sanity check)
            if save_k and idx < save_k:
                try:
                    img_to_save = img[0] if img.ndim == 4 and img.shape[0] == 1 else img
                    ToPILImage()(img_to_save.detach().cpu()).save(os.path.join(save_dir, f"orig{idx+1:03d}.png"))
                except Exception as e:
                    tqdm.write(f"Failed to save original image {idx+1}: {e}")

            # === Feature extraction at fixed t ===
            emb = encoder(img, noise_pct=noise_pct, num_inference_steps=steps, seed=42).float()

            if pool:
                emb = global_avg_pool(emb)
            if emb.ndim == 1:
                emb = emb.unsqueeze(0)

            feats.append(emb.squeeze(0).detach().cpu())

    # Stack + tiny dither to break cosine ties, then L2-normalize
    features = torch.stack(feats).float()  # (N, D)
    features = features + 1e-7 * torch.randn_like(features)  # anti-tie dither
    features = torch.nn.functional.normalize(features, dim=-1)
    return features.cpu().numpy().astype(np.float32)


# ------------------------
# OOO evaluation
# ------------------------

def compute_ooo_acc(features: np.ndarray, triplets: np.ndarray) -> float:
    """
    Compute odd-one-out accuracy given features and triplets.
    Expects helpers.get_predictions to return (choices, ...).
    """
    choices, *_ = helpers.get_predictions(features, triplets, temperature=1.0, dist="cosine")
    return helpers.accuracy(choices)


# ------------------------
# Main
# ------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--things_root", type=str, default="/home/space/thesis/things_starterpack/things_data")
    parser.add_argument("--checkpoint", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--repo_id", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--cc0", action="store_true", help="Use CC0 subset (only if triplets are for CC0).")
    parser.add_argument("--block_idx", type=int, default=8, help="Transformer block index (intermediate often works best).")
    parser.add_argument("--noise_pct", type=float, default=0.20, help="Noise level t in [0..1]. 0.20 = 20%%.")
    parser.add_argument("--steps", type=int, default=100, help="Num inference steps for image2image.")
    parser.add_argument("--num_extractions", type=int, default=None)
    parser.add_argument("--save_k", type=int, default=5)
    args = parser.parse_args()

    # High-level progress across major stages
    stages = [
        "Resolve checkpoint",
        "Initialize encoder",
        "Load dataset",
        "Extract features",
        "Load triplets",
        "Evaluate OOO accuracy",
    ]
    pbar = tqdm(total=len(stages), desc="Pipeline", position=0, leave=True, bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt}")

    # 1) Resolve checkpoint (download if needed)
    ckpt = args.checkpoint
    looks_like_path = ckpt.startswith("/") or ckpt.startswith("./") or ckpt.startswith("../")
    if looks_like_path:
        need_download = (not os.path.exists(ckpt)) or (not os.path.isdir(ckpt)) or (
            not os.path.exists(os.path.join(ckpt, "model_index.json"))
        )
        if need_download:
            os.makedirs(ckpt, exist_ok=True)
            tqdm.write(f"Checkpoint directory '{ckpt}' not found or incomplete. Downloading '{args.repo_id}' to this path...")
            snapshot_download(repo_id=args.repo_id, local_dir=ckpt, local_dir_use_symlinks=False)
            tqdm.write(f"Downloaded model to: {ckpt}")
        ckpt_to_use = ckpt
    else:
        ckpt_to_use = ckpt
    pbar.update(1)

    # 2) Initialize encoder (moves pipeline, sets block)
    encoder = SD3HighestBlockEncoder(checkpoint=ckpt_to_use, block_idx=args.block_idx)
    pbar.update(1)

    # 3) Build THINGS ONCE (aligned=True to match triplet indices) and pass it through
    transform = Compose([Resize((512, 512)), ToTensor()])
    dataset = THINGSBehavior(
        root=args.things_root,
        aligned=True,            # CRITICAL: align image order to triplet indices
        download=False,
        transform=transform,
        cc0=args.cc0,
    )
    pbar.update(1)

    # 4) Extract features at a fixed noise level t
    features = extract_features(
        encoder=encoder,
        dataset=dataset,
        pool=True,
        num_extractions=args.num_extractions,
        noise_pct=float(max(0.0, min(1.0, args.noise_pct))),
        steps=int(max(1, args.steps)),
        save_k=args.save_k,
        save_dir="denoised",
    )
    pbar.update(1)

    # 5) Triplets from the SAME dataset instance
    triplets = dataset.get_triplets()
    try:
        triplets = np.asarray(triplets)
        if triplets.ndim == 2 and triplets.shape[1] >= 2:
            num_feat = features.shape[0]
            # Keep only triplets fully within the extracted subset
            mask = (triplets < num_feat).all(axis=1)
            triplets = triplets[mask]
        else:
            tqdm.write("Triplets format unexpected; skipping filtering.")
    except Exception as e:
        tqdm.write(f"Failed to filter triplets by extracted subset: {e}")
    pbar.update(1)

    # 6) OOO accuracy
    if isinstance(triplets, np.ndarray) and triplets.size > 0:
        acc = compute_ooo_acc(features, triplets)
    else:
        acc = float("nan")
        tqdm.write("No valid triplets after filtering; accuracy is NaN.")
    pbar.update(1)

    pbar.close()
    print(f"Zero-shot odd-one-out accuracy (t={int(100*args.noise_pct)}%): {acc:.4f}")

    # Persist
    try:
        out_dir = "sd3_results"
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir, f"t{int(100*args.noise_pct)}_block{args.block_idx}.txt")
        with open(fname, "w") as f:
            f.write(f"Zero-shot odd-one-out accuracy (t={int(100*args.noise_pct)}%): {acc:.4f}\n")
        tqdm.write(f"Wrote accuracy to '{fname}'")
    except Exception as e:
        tqdm.write(f"Failed to write accuracy file: {e}")


if __name__ == "__main__":
    main()
