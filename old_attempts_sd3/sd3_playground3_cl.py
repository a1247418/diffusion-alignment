# 0.3562% aber nur wenn ohne den expliziten noise 
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

class SD3HighestBlockEncoder(torch.nn.Module):
    def __init__(self, checkpoint: str = "stabilityai/stable-diffusion-3-medium-diffusers", device: str = "cuda", block_idx: Optional[int] = 12):
        super().__init__()
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.checkpoint = checkpoint
        self.block_idx = block_idx
        
        from diffusers import StableDiffusion3Pipeline
        from diffusers.utils import logging as diffusers_logging
        
        diffusers_logging.disable_progress_bar()
        
        # Load the full pipeline to get all components
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            checkpoint,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        
        try:
            self.pipe.set_progress_bar_config(disable=True, leave=False)
        except Exception:
            pass
        
        # Access the transformer directly
        transformer = self.pipe.transformer
        if transformer is None:
            raise ValueError("Could not find transformer in the SD3 pipeline.")
        
        # Write full named_modules map for reference
        try:
            transformer_named = dict(transformer.named_modules())
            with open("transformer_named.txt", "w") as f:
                for key, value in transformer_named.items():
                    f.write(f"{key}: {value}\n")
        except Exception:
            pass
        
        # Get the specific transformer block
        blocks = getattr(transformer, "transformer_blocks", None)
        if blocks is not None and len(blocks) > 0:
            idx = len(blocks) - 1
            if isinstance(self.block_idx, int):
                idx = max(0, min(self.block_idx, len(blocks) - 1))
            self.extraction_module = blocks[idx]
            tqdm.write(f"SD3HighestBlockEncoder: using transformer_blocks.{idx}")
        else:
            raise ValueError("Could not find transformer_blocks")
        
        self.vae = self.pipe.vae
        self.transformer = transformer
        self.generator = torch.Generator(device=self.device)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        use_noise: bool = False,
        noise_level: float = 0.3,
        seed: int = 42,
    ) -> torch.Tensor:
        """
        Extract features by encoding image to latents and passing through transformer.
        
        Args:
            x: Input image tensor [B, C, H, W] in [0, 1]
            use_noise: Whether to add noise to latents before transformer
            noise_level: Amount of noise (0-1) if use_noise=True
            seed: Random seed
        """
        self.generator.manual_seed(seed)
        
        # Encode image to latents
        vae_dtype = self.vae.dtype
        vae_device = next(self.vae.parameters()).device
        
        img_tensor = x.to(device=vae_device, dtype=vae_dtype)
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        # Convert [0,1] to [-1,1] for VAE
        img_tensor = img_tensor * 2.0 - 1.0
        
        # Encode to latents
        scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)
        posterior = self.vae.encode(img_tensor).latent_dist
        latents = posterior.mode()  # Use mode instead of sample for deterministic features
        latents = latents * scaling_factor
        
        # Optionally add controlled noise
        if use_noise and noise_level > 0:
            noise = torch.randn_like(latents, generator=self.generator)
            latents = latents + noise * noise_level
        
        # Prepare for transformer
        # SD3 expects specific input format
        batch_size = latents.shape[0]
        
        # Get empty text embeddings (we're doing unconditional)
        # Create dummy prompts
        prompt_embeds = torch.zeros(
            (batch_size, 77, self.transformer.config.joint_attention_dim),
            device=self.device,
            dtype=latents.dtype
        )
        
        pooled_prompt_embeds = torch.zeros(
            (batch_size, self.transformer.config.pooled_projection_dim),
            device=self.device,
            dtype=latents.dtype
        )
        
        # Use a fixed low timestep for consistent feature extraction
        timestep = torch.tensor([100], device=self.device, dtype=torch.long)
        
        # Hook to capture activations
        activations = []
        
        def hook_fn(_m, _inp, out):
            selected = None
            if isinstance(out, torch.Tensor):
                selected = out
            elif isinstance(out, (list, tuple)):
                if len(out) > 0 and isinstance(out[0], torch.Tensor):
                    selected = out[0]
            elif isinstance(out, dict) and 'hidden_states' in out:
                selected = out['hidden_states']
            
            if selected is not None:
                activations.append(selected.detach())
        
        handle = self.extraction_module.register_forward_hook(hook_fn)
        
        try:
            # Forward pass through transformer
            _ = self.transformer(
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=True,
            )
        finally:
            handle.remove()
        
        if len(activations) == 0:
            raise RuntimeError("No activations were captured from the transformer block.")
        
        return activations[0]


def compute_ooo_acc(features: np.ndarray, triplets: np.ndarray) -> float:
    choices, *_ = helpers.get_predictions(features, triplets, temperature=1.0, dist="cosine")
    return helpers.accuracy(choices)


def extract_features(
    encoder: SD3HighestBlockEncoder,
    things_root: str,
    pool: bool = True,
    cc0: bool = False,
    num_extractions: Optional[int] = None,
    save_k: int = 5,
    save_dir: str = "denoised",
    use_noise: bool = False,
    noise_level: float = 0.3,
) -> np.ndarray:
    transform = Compose([Resize(512), ToTensor()])
    dataset = THINGSBehavior(root=things_root, aligned=False, download=False, transform=transform, cc0=cc0)
    
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)
    effective_total = len(dataset) if (num_extractions is None or num_extractions <= 0) else min(len(dataset), num_extractions)
    
    feats = []
    
    with torch.no_grad():
        if save_k and save_k > 0:
            os.makedirs(save_dir, exist_ok=True)
        
        for idx, sample in enumerate(tqdm(loader, total=effective_total, desc="Extracting features", unit="img")):
            img = sample
            if num_extractions is not None and num_extractions > 0 and idx >= num_extractions:
                break
            
            if isinstance(sample, (list, tuple)):
                img = sample[0]
            
            if save_k and save_k > 0 and idx < save_k:
                try:
                    img_to_save = img[0] if img.ndim == 4 and img.shape[0] == 1 else img
                    pil_img = ToPILImage()(img_to_save.detach().cpu())
                    orig_fname = os.path.join(save_dir, f"orig_{idx+1:03d}.png")
                    pil_img.save(orig_fname)
                except Exception as e:
                    tqdm.write(f"Failed to save original image {idx+1}: {e}")
            
            emb = encoder(img, use_noise=use_noise, noise_level=noise_level)
            
            if pool:
                emb = emb.float()
                # Pool spatial dimensions
                while emb.ndim > 2:
                    emb = emb.mean(dim=-1)
            
            feats.append(emb.squeeze().detach().cpu())
    
    features = torch.stack(feats)
    
    if save_k and save_k > 0:
        try:
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, "features_first_k.txt")
            k = min(save_k, features.shape[0])
            feats_2d = features[:k].view(k, -1).float().cpu().numpy()
            np.savetxt(out_path, feats_2d, fmt="%.6f")
        except Exception as e:
            tqdm.write(f"Failed to save features file: {e}")
    
    return features.numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--things_root", type=str, default="/home/space/thesis/things_starterpack/things_data")
    parser.add_argument("--checkpoint", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--repo_id", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--cc0", action="store_true")
    parser.add_argument("--block_idx", type=int, default=12)
    parser.add_argument("--num_extractions", type=int, default=None)
    parser.add_argument("--save_k", type=int, default=5)
    parser.add_argument("--use_noise", action="store_true", help="Add noise to latents before transformer")
    parser.add_argument("--noise_level", type=float, default=0.3, help="Noise level if use_noise=True")
    args = parser.parse_args()
    
    stages = [
        "Resolve checkpoint",
        "Initialize encoder",
        "Load dataset",
        "Extract features",
        "Load triplets",
        "Evaluate OOO accuracy",
    ]
    
    pbar = tqdm(total=len(stages), desc="Pipeline", position=0, leave=True, bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt}")
    
    # 1) Resolve checkpoint
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
    
    # 2) Initialize encoder
    encoder = SD3HighestBlockEncoder(checkpoint=ckpt_to_use, block_idx=args.block_idx)
    pbar.update(1)
    
    # 3) Load dataset
    transform = Compose([Resize(512), ToTensor()])
    dataset = THINGSBehavior(root=args.things_root, aligned=False, download=False, transform=transform, cc0=args.cc0)
    pbar.update(1)
    
    # 4) Extract features
    features = extract_features(
        encoder,
        things_root=args.things_root,
        pool=True,
        cc0=args.cc0,
        num_extractions=args.num_extractions,
        save_k=args.save_k,
        save_dir="denoised",
        use_noise=args.use_noise,
        noise_level=args.noise_level,
    )
    pbar.update(1)
    
    # 5) Load triplets
    triplets = dataset.get_triplets()
    try:
        triplets = np.asarray(triplets)
        if triplets.ndim == 2 and triplets.shape[1] >= 2:
            num_feat = features.shape[0]
            mask = (triplets < num_feat).all(axis=1)
            triplets = triplets[mask]
        else:
            tqdm.write("Triplets format unexpected; skipping filtering.")
    except Exception as e:
        tqdm.write(f"Failed to filter triplets by extracted subset: {e}")
    
    pbar.update(1)
    
    # 6) Evaluate OOO accuracy
    if isinstance(triplets, np.ndarray) and triplets.size > 0:
        acc = compute_ooo_acc(features, triplets)
    else:
        acc = float("nan")
        tqdm.write("No valid triplets after filtering; accuracy is NaN.")
    
    pbar.update(1)
    pbar.close()
    
    print(f"Zero-shot odd-one-out accuracy: {acc:.4f}")
    
    # Write accuracy
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
    main()