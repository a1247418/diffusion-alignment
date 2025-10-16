#!/bin/bash -l

#SBATCH -a 0-215
#SBATCH -o slurm_logs/sd3_baselines_%A_%a.out
#SBATCH -e slurm_logs/sd3_baselines_%A_%a.err
#SBATCH -D ./
#SBATCH -J sd3_baselines
#SBATCH --partition=gpu-5h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=cuda129
#SBATCH --cpus-per-gpu=14
#SBATCH --mem-per-gpu=100G
#SBATCH --time=05:00:00

mkdir -p slurm_logs

# Activate environment
source test4-sd3/bin/activate

# Set threads (fallback to 14 if SLURM_CPUS_PER_TASK is unset)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-14}

# Paths (adjust if needed)
THINGS_ROOT="/home/space/thesis/things_starterpack/things_data"

# Experiment configurations (3 noises × 3 modes × 24 blocks => array 0..215)
NOISES=(10 50 80)
MODES=(none label description)
NUM_BLOCKS=24

NUM_NOISES=${#NOISES[@]}
NUM_MODES=${#MODES[@]}
TOTAL=$((NUM_BLOCKS * NUM_NOISES * NUM_MODES))

if [ -z "${SLURM_ARRAY_TASK_ID+x}" ]; then
  echo "SLURM_ARRAY_TASK_ID is not set."
  exit 1
fi

TASK_ID=${SLURM_ARRAY_TASK_ID}
if [ "$TASK_ID" -ge "$TOTAL" ]; then
  echo "Task id $TASK_ID out of range (total $TOTAL)."
  exit 1
fi

COMBOS_PER_BLOCK=$(( NUM_NOISES * NUM_MODES ))
BLOCK_IDX=$(( TASK_ID / COMBOS_PER_BLOCK ))
REM=$(( TASK_ID % COMBOS_PER_BLOCK ))
NOISE_IDX=$(( REM / NUM_MODES ))
MODE_IDX=$(( REM % NUM_MODES ))

NOISE=${NOISES[$NOISE_IDX]}
MODE=${MODES[$MODE_IDX]}
NOISE_PCT=$(printf "0.%02d" "$NOISE")
BLOCK=${BLOCK_IDX}

echo "[Array $SLURM_ARRAY_TASK_ID] Starting run at $(date)"
echo "  noise=${NOISE}% (noise_pct=${NOISE_PCT}), text_conditioning=${MODE}, block_idx=${BLOCK}"

# Run (resources are allocated via SBATCH; srun launches the task)
srun python3 sd3_basic_2_not_aligned_text.py \
  --things_root "${THINGS_ROOT}" \
  --noise_pct "${NOISE_PCT}" \
  --text_conditioning "${MODE}" \
  --block_idx "${BLOCK}" \
  --probe

echo "[Array $SLURM_ARRAY_TASK_ID] Finished at $(date)"


