# An analysis of the alignment of latent diffusion models
*Lorenz Linhardt, Marco Morik, Sidney Bender, Naima Elosegui Borras* <br>
Accepted at the ICLR2024 Re-Align workshop. <br>
Paper: https://openreview.net/pdf?id=PFnoxcKh33 <br>

This repository extends the original workshop codebase to also accommodate transformer-based diffusion architectures, namely Stable Diffusion 3 Medium and Stable Diffusion 3.5 Medium. In addition to UNet-based SD1/SD2 models, the pipeline now supports extracting intermediate activations from SD3/SD3.5 transformer blocks, evaluating representational alignment on THINGS (odd-one-out triplets), and optional linear probing. The extension to accommodate transformer models was implemented by Maximilian von Klinski.

The repository offers three entrypoint shell scripts that orchestrate the Python pipeline components:
- `reduction.sh`: dimensionality reduction utilities.
- `regression.sh`: cross-model regression across noise levels.
- `run_all.sh`: end-to-end orchestration of embedding, alignment, and probing over model×module×noise grids (supports SD1/SD2/SD2-Turbo/SD3/SD3.5).

There is also a standalone reference implementation for SD3/SD3.5 transformers in `sd3_basic_2_not_aligned_text_probing.py` which was later integrated into the main pipeline.

```bibtex
@InProceedings{linhardt2024diffusion,
  title={An analysis of the alignment of latent diffusion models},
  author={Lorenz Linhardt and Marco Morik and Sidney Bender and Naima Elosegui Borras},
  year={2024},
  maintitle = {Twelfth International Conference on Learning Representations},
  booktitle = {Workshop on Representational Alignment},
}
```

## Installation & setup

1) Create and activate a Python environment (example shown in the SLURM scripts):
```
source test4-sd3/bin/activate 
# Replace with your own environment with the requirements from requirements_files/requirements_sd3_compatible.txt
```

2) Install dependencies compatible with SD3/SD3.5 and diffusers:
- See `requirements_files/requirements_sd3_compatible.txt` (use your own env if needed).
- Ensure you have a recent `torch`, `diffusers`, `transformers`, `huggingface_hub`, `tqdm`, `numpy`, `torchvision`.
- There is also a requirements file for the thingsvision library, which is only needed if you want to calculate the embeddings for the vision models.

3) Hugging Face model access:
- You may need to accept model licenses and be authenticated for SD3/SD3.5.
- Login once: `huggingface-cli login`.

4) GPU/SLURM environment:
- Scripts assume an HPC setup with SLURM (`srun`) and GPU partitions.
- Adjust partitions, memory, and environment paths to your cluster.

## Data and paths

The pipeline uses the THINGS dataset (behavioral triplets) and derived artifacts:
- `things_root`: path with THINGS images and triplets (e.g., `/home/space/datasets/things` or `/home/space/thesis/things_starterpack/things_data`).
- `data_root`: repository-local workspace for embeddings, alignments, and probing files.
- `probing_root`: root for probing datasets/resources.
- `log_dir`: directory where probing checkpoints and logs are written.
- `path_to_model_dict`: JSON with model metadata for orchestration.

Set these via the shell scripts’ variables or arguments (see below).

## Pipeline overview

The scripts call the following Python modules (names from the original codebase):
- `main_embed.py`:
  - Extracts intermediate features/embeddings for a given model, module, and noise level.
  - For SD1/SD2/SD2-Turbo (UNet), typical modules include `up_blocks.*`, `down_blocks.*`, and `mid_block`.
  - For SD3/SD3.5 (transformers), modules are `transformer_blocks.0` … `transformer_blocks.23`.
- `main_align.py`:
  - Computes representational alignment on THINGS using odd-one-out (OOO) triplets.
  - Supports configurable distance metrics and optional PCA dimensionality reduction.
- `main_probing.py`:
  - Runs linear probing with K-fold CV to learn a transform from features.
  - Logs fold-wise OOO accuracy and can serialize learned transforms.
- `main_reduction.py`:
  - Performs dimensionality reduction utilities (e.g., PCA preparation) over existing features.
- `main_regression.py`:
  - Trains/tests regressions across noise levels and model variants.

The end-to-end flow typically is: embed → align → (optional) probe. The `run_all.sh` script orchestrates this over a grid of models, modules, and noise levels.

## Shell scripts

### `reduction.sh`

Runs dimensionality reduction utilities via SLURM on a single GPU node:

- SLURM header: 1 GPU, 8 CPUs, 80 GB RAM, `gpu-2h` partition.
- Activates environment: `source test4-sd3/bin/activate`.
- Exports `OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}`.
- Key variables:
  - `data_root` (default: `/home/space/datasets/things`)
  - `out_path` (default: `reduction`)
- Command:
```
srun python3 main_reduction.py --data_root $data_root --out_path $out_path
```

Usage notes:
- Customize `data_root` to point to your THINGS data or processed features.
- `out_path` is a folder name for reduced outputs under the working directory.

### `regression.sh`

Runs regression across a noise grid for a base model (CPU-only SLURM job by default):

- SLURM header: CPU partition `cpu-2d`, 8 CPUs, 40 GB RAM.
- Activates environment: `source test4-sd3/bin/activate`.
- Arguments:
  - `-k | --n_folds <int>`: number of folds for cross-validation (default: 5 in the script variable, passed to `main_regression.py` as `--n_folds`).
- Behavior:
  - Uses `base_model="diffusion_stabilityai/stable-diffusion-2-1"`.
  - Builds models for a noise grid `[5,10,20,30,40,50,60,70,80,90]`, resulting in names like `diffusion_stabilityai/stable-diffusion-2-1_20`.
  - Calls:
```
python3 main_regression.py \
  --data_root '/home/space/thesis/things_starterpack/things_data' \
  --n_folds $k \
  --model_names ${models[@]} \
  --load
```

Usage notes:
- Use `-k 5` (or 2–5) to set folds.
- Edit the `base_model` and `noise` array in the script if needed.
- `--load` assumes precomputed features for the listed models are available.

### `run_all.sh`

End-to-end orchestration over a grid of models×modules×noise with SLURM array jobs:

- SLURM header: `#SBATCH -a 0-263` (array covers model×module×noise combinations), `gpu-2d` partition, 1 GPU, 8 CPUs, 80 GB RAM.
- Activates environment: `source test4-sd3/bin/activate`.
- Exports `OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}`.
- Key path variables (customize):
  - `things_root="/home/space/datasets/things"`
  - `data_root="/home/maxvonk/diffusion-alignment/things_playground/things_data"`
  - `path_to_model_dict="/home/space/datasets/things/model_dict_all.json"`
  - `probing_root="/home/maxvonk/diffusion-alignment/things_playground"`
  - `log_dir="/home/maxvonk/diffusion-alignment/things_playground/checkpoints"`
- Toggles and defaults:
  - `base_model="sd3"` (choices: `sd1`, `sd2`, `sd2t`, `sd3`, `sd3.5`).
  - `embed=1`, `align=1`, `probe=1` to enable/disable pipeline stages.
  - `extra_embed="--pool"` (adds `--pool` to embedding stage; can be emptied or extended).
  - `extra_align=""`, `extra_probe=""` for pass-through extra flags.
- CLI options:
  - `-t | --pca <int>`: add `--pca <int>` to align and probe stages; also affects probing dimensionality.
  - `-o | --overwrite`: add `--overwrite` to embed, align, and probe stages to regenerate outputs.
  - `-m | --model {sd1,sd2,sd2t,sd3,sd3.5}`: select base model family.
  - `-p | --prefix {"",conditional,textlast,textlastcapt,conditionalcapt,textlastcapt2,conditionalcapt2,optim,conditionaloptim,optimx1,conditionaloptimx1}`: select conditioning/variant prefix.
  - `-d | --dist <metric>`: set distance metric for alignment (`--distance <metric>`); also appends suffixes to `data_root` and subfolder.
  - `--no_embed` | `--no_align` | `--no_probe`: skip the corresponding stage.
- Prefix validation:
  - The script validates `prefix` against allowed values and aborts otherwise.
- Model mapping and modules:
  - `sd1` → `diffusion_runwayml/stable-diffusion-v1-5`, UNet modules including `up_blocks.*`, `down_blocks.*`, `mid_block`.
  - `sd2` → `diffusion_stabilityai/stable-diffusion-2-1`, UNet modules as above.
  - `sd2t` → `diffusion_stabilityai/sd-turbo`, UNet modules as above.
  - `sd3` → `diffusion_stabilityai/stable-diffusion-3-medium-diffusers`, transformer modules `transformer_blocks.0` … `transformer_blocks.23`.
  - `sd3.5` → `diffusion_stabilityai/stable-diffusion-3.5-medium`, transformer modules `transformer_blocks.0` … `transformer_blocks.23`.
- Prefix-dependent behavior:
  - If `prefix` is one of `{textlast, textlastcapt, textlastcapt2, optim, optimx1}`, the script limits `base_modules=("mid_block")` and `noise=(5)`.
  - Otherwise, a noise grid is used: `noise=(1 5 10 20 30 40 50 60 70 80 90)`.
  - For captioned prefixes, `path_to_caption_dict` is set accordingly (e.g., `caption_dict.npy`, `captionsLavis.npy`, or `optim*` variants).
- Generated grids:
  - `base_models` are created per noise: `<base_model>_<noise>`.
  - The script creates parallel arrays: `models`, `modules`, and `sources` by pairing all model×module combinations; SLURM array index selects a tuple.
- Embedding stage:
```
srun python3 main_embed.py \
  --path_to_caption_dict=${path_to_caption_dict} \
  --path_to_model_dict=${path_to_model_dict} \
  --data_root $data_root \
  --things_root $things_root \
  --model "${models[$SLURM_ARRAY_TASK_ID]}" \
  --module "${modules[$SLURM_ARRAY_TASK_ID]}" \
  --source ${sources[$SLURM_ARRAY_TASK_ID]} \
  ${extra_embed}
```
- Alignment stage:
```
srun python3 main_align.py \
  --data_root $data_root \
  --things_root $things_root \
  --model "${models[$SLURM_ARRAY_TASK_ID]}" \
  --module "${modules[$SLURM_ARRAY_TASK_ID]}" \
  --source "${sources[$SLURM_ARRAY_TASK_ID]}" \
  ${extra_align}
```
- Probing stage:
  - Uses `lambdas=("0.1")` and passes typical training knobs to `main_probing.py`.
  - Example call (simplified):
```
srun python3 main_probing.py \
  --data_root $data_root \
  --dataset things \
  --model "${models[$SLURM_ARRAY_TASK_ID]}" \
  --module "${modules[$SLURM_ARRAY_TASK_ID]}" \
  --source diffusion \
  --lmbda 0.1 \
  --use_bias \
  --probing_root $probing_root \
  --log_dir $log_dir \
  --device gpu \
  --num_processes 8 \
  --subfolder ${subfolder} \
  ${extra_probe}
```

## SD3/SD3.5 standalone reference script

`sd3_basic_2_not_aligned_text_probing.py` is a separate, self-contained implementation targeting the SD3/SD3.5 transformers. It was used as a reference and then integrated into the main pipeline.

Key features:
- Loads `stabilityai/stable-diffusion-3-medium-diffusers` (or 3.5) via `AutoPipelineForImage2Image`.
- Hooks a chosen `transformer_blocks.<idx>` module to capture its first forward output at a fixed noise level.
- Performs optional global average pooling and L2 normalization of features.
- Computes zero-shot odd-one-out (OOO) accuracy on THINGS triplets.
- Optional text conditioning:
  - `--text_conditioning none|label|description`
  - `--guidance_scale` used only when text is provided.
- Optional transformed evaluation:
  - `--apply_transform` can load `.npy/.npz/.pkl` transforms and evaluate OOO accuracy after transforming features.
- Optional probing:
  - `--probe` runs linear probing (K-fold CV) and reports CV accuracy and transformed OOO accuracy.

CLI highlights (subset):
- `--things_root`: THINGS data root.
- `--checkpoint` / `--repo_id`: SD3/SD3.5 model IDs or local paths.
- `--block_idx {0..23}`: transformer block to tap.
- `--noise_pct`: noise level as image2image strength (`[0..1]`).
- `--steps`: inference steps (dense grid so `strength≈t%`).
- `--apply_transform`, `--path_to_transform_features`: for transformed evaluation.
- Probing knobs: `--n_folds`, `--optim`, `--lmbda`, `--batch_size`, `--epochs`, `--use_bias`, `--device`, `--num_processes`.

Example uses:
```
# Zero-shot OOO at t=20%, block 8 (often a strong layer)
python sd3_basic_2_not_aligned_text_probing.py \
  --things_root /path/to/things_data \
  --checkpoint stabilityai/stable-diffusion-3-medium-diffusers \
  --block_idx 8 \
  --noise_pct 0.20

# With description conditioning and probing
python sd3_basic_2_not_aligned_text_probing.py \
  --things_root /path/to/things_data \
  --checkpoint stabilityai/stable-diffusion-3.5-medium \
  --block_idx 12 \
  --noise_pct 0.20 \
  --text_conditioning description \
  --path_to_caption_dict things_playground/things_data/caption_dict.npy \
  --probe --n_folds 3 --lmbda 1e-3 --batch_size 256 --epochs 100 --use_bias
```

## Examples

Run full grid (default SD3, embed+align+probe):
```
sbatch run_all.sh
```

Run SD3.5 with PCA and overwrite existing artifacts:
```
sbatch --export=ALL run_all.sh -m sd3.5 -t 256 -o
```

Skip probing and only embed+align with cosine distance:
```
sbatch run_all.sh -d cosine --no_probe
```

Regression across noise levels with 5-fold CV:
```
sbatch regression.sh -k 5
```

Dimensionality reduction (customize paths inside the script or export env):
```
sbatch reduction.sh
```

## Troubleshooting & tips

- Hugging Face access: accept the SD3/SD3.5 license and run `huggingface-cli login`.
- VRAM: SD3/SD3.5 benefit from GPUs with ≥16 GB. The code selects bf16 if available, otherwise fp16 on GPU, fp32 on CPU for the standalone script.
- SLURM arrays: ensure `#SBATCH -a` range in `run_all.sh` covers the number of model×module combinations you generate.
- Captions: for `*capt*` and `*optim*` prefixes, make sure `path_to_caption_dict` points to existing files.
- Paths: update `things_root`, `data_root`, `probing_root`, `log_dir`, and `path_to_model_dict` for your system.

## Acknowledgements

Some code related to probing and evaluation is adapted from the [human_alignment](https://github.com/LukasMut/human_alignment) and the [gLocal](https://github.com/LukasMut/gLocal) repositories.

