#!/bin/bash -l
#SBATCH -a 0
#SBATCH -o ./reg_%A_%a.out
#SBATCH -e ./reg_%A_%a.err
#SBATCH -D ./
#SBATCH -J regression
#SBATCH --partition="cpu-2d"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40000M

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cogemb2

k=5

while [ ! -z "$1" ];do
  case "$1" in
    -k|--n_folds)
      shift
      k="$1"
      ;;
    *)
    echo "Invalid argument: " $1
  esac
shift
done

base_model="diffusion_stabilityai/stable-diffusion-2-1"
noise=(5 10 20 30 40 50 60 70 80 90)
models=()
for n in "${noise[@]}"; do
    models+=("${base_model}_${n}")
done

echo "Starting regrssion."
echo ${models[@]}
python3 "/home/llorenz/things_playground/main_regression.py" --data_root '/home/llorenz/things_playground/things_data_v2' --n_folds $k --model_names ${models[@]} --load
echo "Finished regression."