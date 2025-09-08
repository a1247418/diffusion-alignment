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
python3 "main_regression.py" --data_root '/home/space/thesis/things_starterpack/things_data' --n_folds $k --model_names ${models[@]} --load
echo "Finished regression."