#!/bin/bash -l
#SBATCH -a 0
#SBATCH -o ./diff_reduce_%A_%a.out
#SBATCH -e ./diff_reduce_%A_%a.err
#SBATCH -D ./
#SBATCH -J reduction
#SBATCH --partition="gpu-2h"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80000M

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cogemb2

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} 

data_root="/home/space/datasets/things";
out_path="/home/llorenz/things_playground/reduction";

device="gpu";

printf "Starting"

srun python3 main_reduction.py --data_root $data_root --out_path $out_path

printf "\nFinished"