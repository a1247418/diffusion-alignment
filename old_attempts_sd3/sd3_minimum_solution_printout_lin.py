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
    def __init__(self, checkpoint: str = "stabilityai/stable-diffusion-3-medium-diffusers", device: str = "cuda"):
        super().__init__()
        # Decide device up front
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.checkpoint = checkpoint

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
            self.extraction_module = list(transformer.modules())[-1]
            # Report which transformer module was selected
            try:
                transformer_named = dict(transformer.named_modules())
                with open("transformer_named.txt", "w") as f:
                    for key, value in transformer_named.items():
                        f.write(f"{key}: {value}\n")
                if len(transformer_named) > 0:
                    sel_name = list(transformer_named.keys())[-1]
                else:
                    sel_name = "(last unnamed)"
            except Exception:
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
            # Some modules output tuples; take the first tensor-like
            if isinstance(out, tuple):
                out = out[0]
            activations.append(out.detach())
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


def extract_features(encoder: SD3HighestBlockEncoder, things_root: str, pool: bool = True, cc0: bool = False) -> np.ndarray:
    transform = Compose([Resize(512), ToTensor()])
    dataset = THINGSBehavior(root=things_root, aligned=False, download=False, transform=transform, cc0=cc0)

    # One clear progress bar for feature extraction over all samples
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)
    feats = []
    with torch.no_grad():
        for sample in tqdm(loader, total=len(dataset), desc="Extracting features", unit="img"):
            img = sample
            if isinstance(sample, (list, tuple)):
                img = sample[0]
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
    encoder = SD3HighestBlockEncoder(checkpoint=ckpt_to_use)
    pbar.update(1)

    # 3) Load dataset (lightweight here, but keep a clear step)
    transform = Compose([Resize(512), ToTensor()])
    dataset = THINGSBehavior(root=args.things_root, aligned=False, download=False, transform=transform, cc0=args.cc0)
    pbar.update(1)

    # 4) Extract features with its own inner progress bar
    features = extract_features(encoder, things_root=args.things_root, pool=True, cc0=args.cc0)
    pbar.update(1)

    # 5) Load triplets
    triplets = dataset.get_triplets()
    pbar.update(1)

    # 6) Evaluate OOO accuracy
    acc = compute_ooo_acc(features, triplets)
    pbar.update(1)
    pbar.close()

    print(f"Zero-shot odd-one-out accuracy: {acc:.4f}")


if __name__ == "__main__":
    # Optional: ensure the classic tqdm look (avoid nested spam)
    # You can also set the env var below from your runner if needed:
    #   os.environ["DIFFUSERS_DISABLE_PROGRESS_BAR"] = "1"
    main()
