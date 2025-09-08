# module purge

# module load cuda/11.2
# module load cudnn/8.1.1.33-11.2

# source test3/bin/activate
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate cogemb2

export OMP_NUM_THREADS=14

data_root="/home/space/datasets/things";
out_path="reduction";

device="gpu";

printf "Starting"

python3 main_reduction.py --data_root $data_root --out_path $out_path

printf "\nFinished"


