import argparse
from typing import Optional, Tuple

import numpy as np
import torch
from torchvision.transforms import Compose, Resize, ToTensor

from things import THINGSBehavior
import helpers
import os
from huggingface_hub import snapshot_download


class SD3HighestBlockEncoder(torch.nn.Module):
    def __init__(self, checkpoint: str = "stabilityai/stable-diffusion-3-medium-diffusers", device: str = "cuda"):
        super().__init__()
        self.device = device if torch.cuda.is_available() else "cpu"
        self.checkpoint = checkpoint
        # Lazy import to keep isolation minimal
        from diffusers import AutoPipelineForImage2Image

        # SD3 uses a DiT-style transformer UNet. Auto pipeline will resolve the right classes if available.
        # We use image2image to mirror existing StableDiffusionEncoder API surface.
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            checkpoint, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

        # Try to locate the highest block in the UNet/transformer backbone
        # Prefer up_blocks last, then mid_block, then down_blocks last.
        unet = getattr(self.pipe, "unet", None)
        if unet is None:
            # Some SD3 repos expose transformer via "transformer" attr
            transformer = getattr(self.pipe, "transformer", None)
            if transformer is None:
                raise ValueError("Could not find UNet/transformer in the SD3 pipeline.")
            # Fallback: take the last block/module
            self.extraction_module = list(transformer.modules())[-1]
        else:
            named = dict(unet.named_modules())
            candidate_names = []
            # up_blocks.*.resnets.1 or attention blocks last
            for i in reversed(range(0, 8)):
                candidate_names.append(f"up_blocks.{i}.resnets.1")
                candidate_names.append(f"up_blocks.{i}.attentions.1.transformer_blocks.0")
            candidate_names.append("mid_block")
            for i in reversed(range(0, 8)):
                candidate_names.append(f"down_blocks.{i}.resnets.1")
                candidate_names.append(f"down_blocks.{i}.attentions.1.transformer_blocks.0")

            found = None
            for name in candidate_names:
                if name in named:
                    found = named[name]
                    break
            if found is None:
                # Fallback to the very last named module
                found = list(named.values())[-1]
            self.extraction_module = found

        self.generator = torch.Generator(device=self.device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, strength: float = 0.9, num_inference_steps: int = 20, seed: int = 42, stop_early: bool = True) -> torch.Tensor:
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

    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)
    feats = []
    with torch.no_grad():
        for sample in loader:
            img = sample
            if isinstance(sample, (list, tuple)):
                img = sample[0]
            emb = encoder(img, stop_early=True)
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

    # If checkpoint is a local path that doesn't exist yet, download repo to this path
    ckpt = args.checkpoint
    looks_like_path = ckpt.startswith("/") or ckpt.startswith("./") or ckpt.startswith("../")
    if looks_like_path:
        need_download = (not os.path.exists(ckpt)) or (not os.path.isdir(ckpt)) or (not os.path.exists(os.path.join(ckpt, "model_index.json")))
        if need_download:
            os.makedirs(ckpt, exist_ok=True)
            print(f"Checkpoint directory '{ckpt}' not found or incomplete. Downloading '{args.repo_id}' to this path...")
            snapshot_download(repo_id=args.repo_id, local_dir=ckpt, local_dir_use_symlinks=False)
            print(f"Downloaded model to: {ckpt}")
        ckpt_to_use = ckpt
    else:
        # Treat as repo id or resolved cache
        ckpt_to_use = ckpt

    encoder = SD3HighestBlockEncoder(checkpoint=ckpt_to_use)

    # Extract features
    features = extract_features(encoder, things_root=args.things_root, pool=True, cc0=args.cc0)

    # Compute zero-shot odd-one-out accuracy using existing helpers
    transform = Compose([Resize(512), ToTensor()])
    dataset = THINGSBehavior(root=args.things_root, aligned=False, download=False, transform=transform, cc0=args.cc0)
    triplets = dataset.get_triplets()
    acc = compute_ooo_acc(features, triplets)
    print("Zero-shot odd-one-out accuracy:", acc)


if __name__ == "__main__":
    main()
