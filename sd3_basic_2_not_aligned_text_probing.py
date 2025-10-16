import argparse
from typing import Optional
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
# Transforms / Probing helpers
# ------------------------

def _apply_transform_matrix(features: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Apply a linear transform to features. If transform has a bias column, use it.
    Mirrors conventions in probing where features are standardized by global mean/std first.
    """
    x = features.astype(np.float32, copy=True)

    if transform.ndim == 2 and transform.shape[1] == x.shape[1] + 1:
        # last column is bias
        weights = transform[:, :-1]
        bias = transform[:, -1]
        y = x @ weights.T + bias[None, :]
    elif transform.ndim == 2 and transform.shape[0] == x.shape[1]:
        # weights only (D, D)
        y = x @ transform
    else:
        # Try the other orientation if saved as (D_out, D_in)
        if transform.ndim == 2 and transform.shape[1] == x.shape[1]:
            y = x @ transform.T
        else:
            raise ValueError(f"Transform shape {transform.shape} incompatible with features {x.shape}.")
    # L2-normalize rows to ensure cosine matches paper
    norms = np.linalg.norm(y, axis=1, keepdims=True) + 1e-8
    y = y / norms
    return y.astype(np.float32)


def _apply_transform_from_path(features: np.ndarray, path: str, *, path_to_features_for_global: Optional[str] = None) -> np.ndarray:
    """
    Load a transform from .npy/.npz/.pkl and apply to features.
    - .npy: raw matrix (optionally with bias as last column)
    - .npz: expects keys like 'weights','bias','mean','std' (glocal) or a raw array under default key
    - .pkl: expects structure like in transform.GlobalTransform; requires path_to_features_for_global
    """
    path = str(path)
    if path.endswith(".npy"):
        t = np.load(path)
        return _apply_transform_matrix(features, t)
    elif path.endswith(".npz"):
        data = np.load(path)
        if isinstance(data, np.lib.npyio.NpzFile):
            keys = set(list(data.keys()))
            if {"weights", "mean", "std"}.issubset(keys):
                x = (features - data["mean"]) / (data["std"] + 1e-8)
                y = x @ data["weights"]
                if "bias" in keys:
                    y = y + data["bias"][None, :]
                return y.astype(np.float32)
            elif len(keys) == 1 and "arr_0" in keys:
                t = data["arr_0"]
                return _apply_transform_matrix(features, t)
            else:
                # Fallback: try treating the first array as weights
                first_key = sorted(keys)[0]
                t = data[first_key]
                return _apply_transform_matrix(features, t)
        else:
            # Unexpected, but attempt raw array semantics
            return _apply_transform_matrix(features, np.array(data))
    elif path.endswith(".pkl"):
        # Use GlobalTransform-style loader
        from transform import GlobalTransform  # lazy import
        if not path_to_features_for_global:
            raise ValueError("Applying .pkl transform requires --path_to_transform_features for mean/std.")
        gt = GlobalTransform(
            path_to_transform=path,
            path_to_features=path_to_features_for_global,
        )
        return gt.transform_features(features)
    else:
        raise ValueError(f"Unsupported transform file extension for '{path}'.")


def _apply_learned_transform(features: np.ndarray, transform_obj) -> np.ndarray:
    """
    Apply a learned transform object (from probing.run) to features and L2-normalize rows.
    Supports:
      - numpy arrays (optionally with bias in last column)
      - torch tensors
      - dicts with keys like 'weights','bias','mean','std'
      - objects exposing transform_features(features)
    """
    x = features.astype(np.float32, copy=False)

    # Torch -> numpy
    try:
        import torch as _torch  # local import to avoid polluting namespace
        if isinstance(transform_obj, _torch.Tensor):
            transform_obj = transform_obj.detach().cpu().numpy()
    except Exception:
        pass

    # Direct array matrix application (and optional bias)
    if isinstance(transform_obj, np.ndarray):
        return _apply_transform_matrix(x, transform_obj)

    # Dict-based saved parameters
    if isinstance(transform_obj, dict):
        weights = transform_obj.get("weights", None)
        bias = transform_obj.get("bias", None)
        mean = transform_obj.get("mean", None)
        std = transform_obj.get("std", None)

        if mean is not None and std is not None:
            x = (x - np.asarray(mean, dtype=np.float32)) / (np.asarray(std, dtype=np.float32) + 1e-8)

        if weights is not None:
            w = np.asarray(weights, dtype=np.float32)
            if w.ndim != 2:
                raise ValueError("'weights' in transform must be 2D")
            if w.shape[0] == x.shape[1]:
                y = x @ w
            elif w.shape[1] == x.shape[1]:
                y = x @ w.T
            else:
                raise ValueError(f"Weights shape {w.shape} incompatible with features {x.shape}.")
        else:
            # Fallback: try first array-like entry
            arr_keys = [k for k, v in transform_obj.items() if isinstance(v, (np.ndarray, list, tuple))]
            if not arr_keys:
                raise ValueError("Transform dict does not contain weights.")
            w = np.asarray(transform_obj[arr_keys[0]], dtype=np.float32)
            return _apply_transform_matrix(x, w)

        if bias is not None:
            b = np.asarray(bias, dtype=np.float32)
            if b.ndim == 1:
                y = y + b[None, :]
            else:
                y = y + b

        # L2-normalize rows
        norms = np.linalg.norm(y, axis=1, keepdims=True) + 1e-8
        y = y / norms
        return y.astype(np.float32)

    # Method-based transform
    if hasattr(transform_obj, "transform_features") and callable(getattr(transform_obj, "transform_features")):
        y = transform_obj.transform_features(x)
        # ensure numpy and float32
        y = np.asarray(y, dtype=np.float32)
        norms = np.linalg.norm(y, axis=1, keepdims=True) + 1e-8
        y = y / norms
        return y.astype(np.float32)

    raise ValueError("Unsupported learned transform object type.")


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

        # Build pipeline, prefer bf16 on CUDA if available; otherwise fp16 on CUDA, fp32 on CPU
        if self.device == "cuda":
            preferred_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            preferred_dtype = torch.float32
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            checkpoint,
            torch_dtype=preferred_dtype,
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
        prompt: Optional[str] = "",
        guidance_scale: float = 3.5,
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
                    prompt=prompt if isinstance(prompt, str) else "",
                    guidance_scale=float(guidance_scale),
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
    text_conditioning: str = "none",  # one of: none, label, description
    captions_dict: Optional[dict] = None,
    guidance_scale: float = 3.5,
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
            prompt: Optional[str] = None
            if text_conditioning in ("label", "description"):
                # sample may be (img_batch, names) with batch_size=1
                try:
                    name_field = sample[1]
                    name_value = name_field[0] if isinstance(name_field, (list, tuple)) else name_field
                    if isinstance(name_value, (np.ndarray, torch.Tensor)):
                        # Convert single-element tensors/arrays to Python string
                        try:
                            name_value = name_value.item()
                        except Exception:
                            pass
                    if text_conditioning == "label":
                        concept = str(name_value).split(".")[0].replace("_", " ")
                        # remove digits, mirror original implementation
                        concept = "".join([ch for ch in concept if not ch.isdigit()])
                        prompt = f"a photo of a {concept}".strip()
                    elif text_conditioning == "description":
                        if captions_dict is None:
                            raise ValueError("text_conditioning='description' requires a captions dict.")
                        prompt = captions_dict[str(name_value)]
                except Exception:
                    prompt = None

            # Use guidance only when we actually condition on text
            gs = float(guidance_scale) if (isinstance(prompt, str) and len(prompt) > 0) else 0.0

            emb = (
                encoder(
                    img,
                    noise_pct=noise_pct,
                    num_inference_steps=steps,
                    seed=42,
                    prompt=prompt if isinstance(prompt, str) else "",
                    guidance_scale=gs,
                ).float()
            )

            if pool:
                emb = global_avg_pool(emb)
            # Ensure per-sample embedding is 1D (D,)
            if emb.ndim >= 3:
                emb = emb.view(emb.shape[0], -1)
            if emb.ndim == 2:
                # squeeze batch dim if present
                if emb.shape[0] == 1:
                    emb = emb.squeeze(0)
                else:
                    # average across first dim to obtain a single vector
                    emb = emb.mean(dim=0)
            elif emb.ndim == 1:
                pass
            else:
                emb = emb.view(-1)

            feats.append(emb.detach().cpu())

    # Stack, coerce to (N, D), then L2-normalize
    features = torch.stack(feats).float()
    if features.ndim > 2:
        features = features.view(features.shape[0], -1)
    elif features.ndim == 1:
        features = features.unsqueeze(0)
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
    # Text conditioning
    parser.add_argument(
        "--text_conditioning",
        type=str,
        default="none",
        choices=["none", "label", "description"],
        help="Text conditioning: none, label, or description.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="Guidance scale for text conditioning (ignored if none).",
    )
    parser.add_argument(
        "--path_to_caption_dict",
        type=str,
        default="things_playground/things_data/caption_dict.npy",
        help="Path to captions dict (.npy) when using description conditioning.",
    )
    # Transformed evaluation / probing
    parser.add_argument("--apply_transform", type=str, default=None, help="Path to saved transform (.npy/.npz/.pkl) to apply before evaluation.")
    parser.add_argument("--path_to_transform_features", type=str, default=None, help="Required when --apply_transform is .pkl: path to features pickle for mean/std.")
    parser.add_argument("--probe", action="store_true", help="Run linear probing to learn a transform and evaluate.")
    parser.add_argument("--probing_root", type=str, default="sd3_probing", help="Root dir for probing artifacts/logs.")
    parser.add_argument("--log_dir", type=str, default="sd3_probing/checkpoints", help="Directory to checkpoint probing.")
    parser.add_argument("--n_folds", type=int, default=3, choices=[2,3,4,5], help="Folds for K-fold probing.")
    parser.add_argument("--optim", type=str, default="Adam", choices=["Adam","AdamW","SGD"], help="Optimizer for probing.")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lmbda", type=float, default=1e-3, choices=[10.0,1.0,1e-1,1e-2,1e-3,1e-4,1e-5])
    parser.add_argument("--batch_size", type=int, default=256, choices=[64,128,256,512,1024])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--burnin", type=int, default=3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--use_bias", action="store_true")
    parser.add_argument("--device", type=str, default="gpu", choices=["cpu","gpu"], help="Device for probing trainer.")
    parser.add_argument("--num_processes", type=int, default=8, help="Num devices for CPU distributed probing.")
    parser.add_argument("--rnd_seed", type=int, default=42)
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
    if args.apply_transform:
        stages.append("Apply transform + evaluate")
    if args.probe:
        stages.append("Probing + evaluate")
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
    need_names = args.text_conditioning in ("label", "description")
    dataset = THINGSBehavior(
        root=args.things_root,
        aligned=True,            # CRITICAL: align image order to triplet indices
        download=False,
        transform=transform,
        cc0=args.cc0,
        return_names=need_names,
    )
    pbar.update(1)

    # 4) Extract features at a fixed noise level t
    # Load captions dictionary if needed
    captions_dict = None
    if args.text_conditioning == "description" and args.path_to_caption_dict:
        try:
            cap = np.load(args.path_to_caption_dict, allow_pickle=True)
            try:
                cap = cap[()]
            except Exception:
                pass
            captions_dict = cap
        except Exception as e:
            tqdm.write(f"Failed to load captions dict '{args.path_to_caption_dict}': {e}")

    features = extract_features(
        encoder=encoder,
        dataset=dataset,
        pool=True,
        num_extractions=args.num_extractions,
        noise_pct=float(max(0.0, min(1.0, args.noise_pct))),
        steps=int(max(1, args.steps)),
        save_k=args.save_k,
        save_dir="denoised",
        text_conditioning=args.text_conditioning,
        captions_dict=captions_dict,
        guidance_scale=args.guidance_scale,
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

    # Optional: transformed evaluation using provided transform file
    if args.apply_transform:
        try:
            feats_tx = _apply_transform_from_path(
                features,
                args.apply_transform,
                path_to_features_for_global=args.path_to_transform_features,
            )
            acc_tx = compute_ooo_acc(feats_tx, triplets)
            tqdm.write(f"Transformed odd-one-out accuracy: {acc_tx:.4f}")
        except Exception as e:
            acc_tx = float("nan")
            tqdm.write(f"Failed transformed evaluation: {e}")
        pbar.update(1)
    else:
        acc_tx = None

    # Optional: probing to learn a transform and evaluate
    avg_cv_acc = None
    acc_tx_probe = None
    if args.probe:
        try:
            # Lazy import probing utilities
            import main_probing as probing

            # Build minimal args-like object for probing config
            class _Obj:
                pass
            pr_args = _Obj()
            pr_args.optim = args.optim
            pr_args.learning_rate = args.learning_rate
            pr_args.lmbda = args.lmbda
            pr_args.n_folds = args.n_folds
            pr_args.batch_size = args.batch_size
            pr_args.epochs = args.epochs
            pr_args.burnin = args.burnin
            pr_args.patience = args.patience
            pr_args.use_bias = args.use_bias
            pr_args.model = [f"sd3_block{args.block_idx}_t{int(100*args.noise_pct)}"]
            pr_args.module = ["highest_block"]
            pr_args.log_dir = args.log_dir

            optim_cfg = probing.create_optimization_config(pr_args)

            ooo_choices, cv_results, transform = probing.run(
                features=features,
                data_root=args.things_root,
                n_objects=features.shape[0],
                device=args.device,
                optim_cfg=optim_cfg,
                rnd_seed=args.rnd_seed,
                num_processes=args.num_processes,
                pca=None,
            )
            # Report fold-wise cross-validated accuracy
            avg_cv_acc = probing.get_mean_cv_acc(cv_results)
            tqdm.write(f"Probing cross-validated accuracy (fold-wise OOO): {avg_cv_acc:.4f}")

            # Apply learned transform to the current features to get transformed accuracy,
            # comparable to the pipeline's final transformed accuracy reporting.
            try:
                feats_tx_probe = _apply_learned_transform(features, transform)
                acc_tx_probe = compute_ooo_acc(feats_tx_probe, triplets)
                tqdm.write(f"Probing transformed odd-one-out accuracy: {acc_tx_probe:.4f}")
            except Exception as e:
                acc_tx_probe = float("nan")
                tqdm.write(f"Failed to compute probing transformed accuracy: {e}")
        except Exception as e:
            tqdm.write(f"Probing failed: {e}")
        pbar.update(1)

    pbar.close()
    print(f"Zero-shot odd-one-out accuracy (t={int(100*args.noise_pct)}%): {acc:.4f}")
    if acc_tx is not None and not np.isnan(acc_tx):
        print(f"Transformed odd-one-out accuracy: {acc_tx:.4f}")
    try:
        if 'acc_tx_probe' in locals() and acc_tx_probe is not None and not np.isnan(acc_tx_probe):
            print(f"Probing transformed odd-one-out accuracy: {acc_tx_probe:.4f}")
    except Exception:
        pass

    # Persist
    try:
        base_dir = "sd3_results"
        dest_dirs = [base_dir]
        # Also write into probing-specific subfolders when probing/transform is used
        if args.probe or args.apply_transform:
            dest_dirs.append(os.path.join(base_dir, "probing"))
            if isinstance(args.text_conditioning, str) and args.text_conditioning != "none":
                dest_dirs.append(os.path.join(base_dir, f"probing_{args.text_conditioning}"))

        for out_dir in dest_dirs:
            os.makedirs(out_dir, exist_ok=True)
            fname = os.path.join(out_dir, f"t{int(100*args.noise_pct)}_block{args.block_idx}.txt")
            with open(fname, "w") as f:
                f.write(f"Zero-shot odd-one-out accuracy (t={int(100*args.noise_pct)}%): {acc:.4f}\n")
                if acc_tx is not None and not np.isnan(acc_tx):
                    f.write(f"Transformed odd-one-out accuracy: {acc_tx:.4f}\n")
                if acc_tx_probe is not None and not np.isnan(acc_tx_probe):
                    f.write(f"Probing transformed odd-one-out accuracy: {acc_tx_probe:.4f}\n")
                if avg_cv_acc is not None and not np.isnan(avg_cv_acc):
                    f.write(f"Probing cross-validated accuracy (fold-wise OOO): {avg_cv_acc:.4f}\n")
            tqdm.write(f"Wrote accuracy to '{fname}'")
    except Exception as e:
        tqdm.write(f"Failed to write accuracy file: {e}")


if __name__ == "__main__":
    main()
