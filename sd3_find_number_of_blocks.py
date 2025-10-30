# pip install --upgrade diffusers transformers accelerate torch safetensors huggingface_hub
import json
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, snapshot_download

from diffusers import AutoPipelineForText2Image
from diffusers.utils import logging

logging.set_verbosity_error()

MODELS = [
    "stabilityai/stable-diffusion-3-medium-diffusers",
    "stabilityai/stable-diffusion-3.5-medium",
    "stabilityai/stable-diffusion-3.5-large",
]

def get_block_count_from_transformer(transformer) -> int | None:
    """
    Try several common patterns used by Diffusers' transformer modules.
    Returns an int if we can confidently count, else None.
    """
    # 1) The most reliable: a ModuleList of top-level transformer layers
    for attr in ("transformer_blocks", "layers", "blocks"):
        if hasattr(transformer, attr):
            val = getattr(transformer, attr)
            try:
                return len(val)  # Works if it's a list/ModuleList
            except TypeError:
                pass

    # 2) Many configs expose num_layers
    cfg = getattr(transformer, "config", None)
    if cfg is not None:
        for k in ("num_layers", "n_layer", "layers", "depth", "transformer_layers"):
            if hasattr(cfg, k):
                v = getattr(cfg, k)
                if isinstance(v, int) and v > 0:
                    return v

    # 3) Conservative heuristic: count top-level child modules with "Block" in their class name
    #    (Avoids overcounting nested sub-blocks in attention/MLP)
    try:
        top_level_children = list(transformer.children())
        block_like = [m for m in top_level_children if "Block" in m.__class__.__name__]
        if block_like:
            return len(block_like)
    except Exception:
        pass

    return None


def try_config_num_layers_only(repo_id: str) -> int | None:
    """
    If loading the full pipeline fails, fetch transformer config and see if there's a num_layers-like field.
    """
    try:
        # Either the transformer sits under subfolder "transformer", or config is at top-level
        # We’ll try both.
        tmp_dir = snapshot_download(repo_id, allow_patterns=["**/config.json", "config.json"])
        candidates = [
            Path(tmp_dir) / "transformer" / "config.json",
            Path(tmp_dir) / "config.json",
        ]
        for c in candidates:
            if c.exists():
                with open(c, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                for k in ("num_layers", "n_layer", "layers", "depth", "transformer_layers"):
                    v = cfg.get(k)
                    if isinstance(v, int) and v > 0:
                        return v
    except Exception:
        pass
    return None


def count_blocks(repo_id: str) -> int | None:
    """
    Load the model (pipeline), access its transformer, and count blocks.
    Falls back to config-only approach if needed.
    """
    print(f"\n=== {repo_id} ===")
    # Step 1: try to load the pipeline (the usual entrypoint in Diffusers)
    pipe = None
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(
            repo_id,
            torch_dtype=torch.float16,
            variant="fp16",              # many SD pipelines publish an fp16 variant
            trust_remote_code=True,      # SD 3.x / 3.5 may require custom components
        )
    except Exception as e:
        print(f"Pipeline load failed (will try config-only): {e}")

    if pipe is not None:
        # Try to find the transformer module on the pipeline
        transformer = getattr(pipe, "transformer", None)
        if transformer is None:
            # Some pipelines name it differently; try common fallbacks
            for name in ("unet", "dit", "mmdit", "flux_transformer", "transformer_model"):
                if hasattr(pipe, name):
                    transformer = getattr(pipe, name)
                    break

        if transformer is None:
            print("Could not locate a transformer module on the pipeline.")
        else:
            n = get_block_count_from_transformer(transformer)
            if isinstance(n, int) and n > 0:
                return n
            else:
                print("Could not directly infer count from transformer; trying config…")

    # Step 2: try config-based fallback (no heavy model weights)
    n_cfg = try_config_num_layers_only(repo_id)
    return n_cfg


def main():
    results = {}
    for repo in MODELS:
        count = count_blocks(repo)
        results[repo] = count

    print("\n--- Results ---")
    for repo, n in results.items():
        if n is None:
            print(f"{repo}: (couldn’t determine from public artifacts)")
        else:
            print(f"{repo}: {n} transformer blocks")

if __name__ == "__main__":
    main()
