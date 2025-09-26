import argparse
from typing import Optional, Tuple

import os
import numpy as np
import torch
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm

from things import THINGSBehavior
import helpers
from huggingface_hub import snapshot_download


class SD3HighestBlockEncoder(torch.nn.Module):
    def __init__(self, checkpoint: str = "stabilityai/stable-diffusion-3-medium-diffusers", device: str = "cuda", block_idx: Optional[int] = 12):
        super().__init__()
        # Decide device up front
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.checkpoint = checkpoint
        self.block_idx = block_idx

        # --- Diffusers imports kept lazy to keep deps isolated
        from diffusers import AutoPipelineForImage2Image
        from diffusers.utils import logging as diffusers_logging

        # Disable all internal diffusers progress bars globally
        diffusers_logging.disable_progress_bar()

        # Build pipeline (no internal tqdm)
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            checkpoint,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

        # Extra safety: also turn off pipeline’s own progress bar just in case
        try:
            self.pipe.set_progress_bar_config(disable=True, leave=False)
        except Exception:
            pass

        # Try to locate the highest block in the UNet/transformer backbone
        unet = getattr(self.pipe, "unet", None)
        if unet is None:
            transformer = getattr(self.pipe, "transformer", None)
            if transformer is None:
                raise ValueError("Could not find UNet/transformer in the SD3 pipeline.")
            # Write full named_modules map for reference
            try:
                transformer_named = dict(transformer.named_modules())
                with open("transformer_named.txt", "w") as f:
                    for key, value in transformer_named.items():
                        f.write(f"{key}: {value}\n")
            except Exception:
                transformer_named = None

            # Prefer the requested transformer block if available; fallback to last
            sel_name = "(last unnamed)"
            block_module = None
            try:
                blocks = getattr(transformer, "transformer_blocks", None)
                if blocks is not None and len(blocks) > 0:
                    idx = len(blocks) - 1
                    if isinstance(self.block_idx, int):
                        # clamp to valid range
                        idx = max(0, min(self.block_idx, len(blocks) - 1))
                    block_module = blocks[idx]
                    sel_name = f"transformer_blocks.{idx}"
            except Exception:
                block_module = None

            if block_module is not None:
                self.extraction_module = block_module
            else:
                # Fallback: use last registered module
                self.extraction_module = list(transformer.modules())[-1]
                if transformer_named is not None and len(transformer_named) > 0:
                    try:
                        sel_name = list(transformer_named.keys())[-1]
                    except Exception:
                        sel_name = "(last unnamed)"
                else:
                    sel_name = "(last unnamed)"

            tqdm.write(
                f"SD3HighestBlockEncoder: using module 'transformer.{sel_name}' ({type(self.extraction_module).__name__})"
            )
        else:
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
        strength: float = 0.1,
        num_inference_steps: int = 20,
        seed: int = 42,
        stop_early: bool = True,
    ) -> torch.Tensor:
        # Ensure any pipeline progress bars stay disabled on call
        try:
            self.pipe.set_progress_bar_config(disable=True, leave=False)
        except Exception:
            pass

        self.generator.manual_seed(seed)
        activations = []

        def hook_fn(_m, _inp, out):
            # Select a meaningful tensor from various output structures
            selected = None
            if isinstance(out, torch.Tensor):
                selected = out
            elif isinstance(out, (list, tuple)):
                # Prefer second element when present (often hidden_states), else first tensor
                if len(out) > 1 and isinstance(out[1], torch.Tensor):
                    selected = out[1]
                else:
                    for elem in out:
                        if isinstance(elem, torch.Tensor):
                            selected = elem
                            break
            elif isinstance(out, dict):
                for v in out.values():
                    if isinstance(v, torch.Tensor):
                        selected = v
                        break

            if selected is not None:
                activations.append(selected.detach())
                if stop_early:
                    raise _EarlyStop

        handle = self.extraction_module.register_forward_hook(hook_fn)
        try:
            try:
                self.pipe(
                    image=x.to(self.device),
                    strength=strength,
                    prompt="",
                    guidance_scale=0.0,
                    num_inference_steps=num_inference_steps,
                    generator=self.generator,
                )
            except _EarlyStop:
                pass
        finally:
            handle.remove()

        if len(activations) == 0:
            raise RuntimeError("No activations were captured from the highest block.")
        return activations[0]


class _EarlyStop(Exception):
    pass


def compute_ooo_acc(features: np.ndarray, triplets: np.ndarray) -> float:
    choices, _ = helpers.get_predictions(features, triplets, temperature=1.0, dist="cosine")
    return helpers.accuracy(choices)


def extract_features(
    encoder: SD3HighestBlockEncoder,
    things_root: str,
    pool: bool = True,
    cc0: bool = False,
    save_k: int = 5,
    save_dir: str = "denoised",
) -> np.ndarray:
    transform = Compose([Resize(512), ToTensor()])
    dataset = THINGSBehavior(root=things_root, aligned=False, download=False, transform=transform, cc0=cc0)

    # One clear progress bar for feature extraction over all samples
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)
    feats = []
    with torch.no_grad():
        if save_k and save_k > 0:
            os.makedirs(save_dir, exist_ok=True)

        for idx, sample in enumerate(tqdm(loader, total=len(dataset), desc="Extracting features", unit="img")):
            img = sample
            if isinstance(sample, (list, tuple)):
                img = sample[0]
            if save_k and save_k > 0 and idx < save_k:
                try:
                    try:
                        encoder.pipe.set_progress_bar_config(disable=True, leave=False)
                    except Exception:
                        pass
                    result = encoder.pipe(
                        image=img.to(encoder.device),
                        strength=0.1,
                        prompt="",
                        guidance_scale=0.0,
                        num_inference_steps=20,
                        generator=encoder.generator,
                    )
                    if hasattr(result, "images") and len(result.images) > 0:
                        out_img = result.images[0]
                        fname = os.path.join(save_dir, f"{idx+1:03d}.png")
                        out_img.save(fname)
                except Exception as e:
                    tqdm.write(f"Failed to save denoised image {idx+1}: {e}")
            emb = encoder(img, stop_early=True)  # internal bars already disabled
            if pool:
                emb = emb.float()
                while emb.ndim > 2:
                    emb = emb.mean(dim=-1)
            feats.append(emb.squeeze().detach().cpu())
    features = torch.stack(feats)
    return features.numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--things_root", type=str, default="/home/space/thesis/things_starterpack/things_data")
    parser.add_argument("--checkpoint", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--repo_id", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--cc0", action="store_true")
    parser.add_argument("--block_idx", type=int, default=12)
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
        need_download = (not os.path.exists(ckpt)) or (not os.path.isdir(ckpt)) or (not os.path.exists(os.path.join(ckpt, "model_index.json")))
        if need_download:
            os.makedirs(ckpt, exist_ok=True)
            tqdm.write(f"Checkpoint directory '{ckpt}' not found or incomplete. Downloading '{args.repo_id}' to this path...")
            snapshot_download(repo_id=args.repo_id, local_dir=ckpt, local_dir_use_symlinks=False)
            tqdm.write(f"Downloaded model to: {ckpt}")
        ckpt_to_use = ckpt
    else:
        ckpt_to_use = ckpt
    pbar.update(1)

    # 2) Initialize encoder (this constructs and moves the pipeline)
    encoder = SD3HighestBlockEncoder(checkpoint=ckpt_to_use, block_idx=args.block_idx)
    pbar.update(1)

    # 3) Load dataset (lightweight here, but keep a clear step)
    transform = Compose([Resize(512), ToTensor()])
    dataset = THINGSBehavior(root=args.things_root, aligned=False, download=False, transform=transform, cc0=args.cc0)
    pbar.update(1)

    # 4) Extract features with its own inner progress bar
    features = extract_features(
        encoder,
        things_root=args.things_root,
        pool=True,
        cc0=args.cc0,
        save_k=args.save_k,
        save_dir="denoised",
    )
    pbar.update(1)

    # 5) Load triplets
    triplets = dataset.get_triplets()
    pbar.update(1)

    # 6) Evaluate OOO accuracy
    acc = compute_ooo_acc(features, triplets)
    pbar.update(1)
    pbar.close()

    print(f"Zero-shot odd-one-out accuracy: {acc:.4f}")

    # Write accuracy to sd3_results/<block_idx>.txt
    try:
        out_dir = "sd3_results"
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir, f"{args.block_idx}.txt")
        with open(fname, "w") as f:
            f.write(f"Zero-shot odd-one-out accuracy: {acc:.4f}\n")
        tqdm.write(f"Wrote accuracy to '{fname}'")
    except Exception as e:
        tqdm.write(f"Failed to write accuracy file: {e}")


if __name__ == "__main__":
    # Optional: ensure the classic tqdm look (avoid nested spam)
    # You can also set the env var below from your runner if needed:
    #   os.environ["DIFFUSERS_DISABLE_PROGRESS_BAR"] = "1"
    main()
