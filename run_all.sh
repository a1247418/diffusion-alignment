#!/bin/bash -l
#SBATCH -a 0
#SBATCH -o ./all_%A_%a.out
#SBATCH -e ./all_%A_%a.err
#SBATCH -D ./
#SBATCH -J all
#SBATCH --partition="gpu-2d"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80000M

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cogemb2

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} 

dataset="things";
device="gpu";

things_root="/home/space/datasets/things";
data_root="/home/llorenz/things_playground/things_data";
path_to_model_dict="/home/space/datasets/things/model_dict_all.json"
probing_root="/home/llorenz/things_playground";
log_dir="/home/llorenz/things_playground/checkpoints";

extra_embed=""
extra_align=""
extra_probe=""
base_model="sd2t"
subfolder="things_data"
embed=1
align=1
probe=1

prefix="" #nothing
# prefix="conditional" #class-label conditional
# prefix="textlast" #text embedding of class label
# prefix="textlastcapt" #text embedding of caption
# prefix="conditionalcapt" #caption conditional
# prefix="textlastcapt2"  #text embedding of caption2
# prefix="conditionalcapt2" #caption2 conditional
# prefix="optim" # optimitzed embedding
# prefix="conditionaloptim" # optimitzed-embedding conditional
# prefix="optimx1" # optimitzed embedding
# prefix="conditionaloptimx1" # optimitzed-embedding conditional

while [ ! -z "$1" ];do
  case "$1" in
    -t|--pca)
	shift
	extra_align=${extra_align}" --pca $1"
	extra_probe=${extra_probe}" --pca $1"
      ;;
    -o|--overwrite)
	extra_embed=${extra_embed}" --overwrite"
	extra_align=${extra_align}" --overwrite"
	extra_probe=${extra_probe}" --overwrite"
      ;;
    -m|--model)
	shift
	base_model="$1"
      ;;
    -p|--prefix)
	shift
	prefix="$1"
      ;;	
    -v|--v2)
	extra_embed=${extra_embed}" --v2"
	data_root=${data_root}"_v2"
	subfolder="things_data_v2"
      ;;
    -d|--dist)
	shift
	dist="$1"
	extra_align=${extra_align}" --distance "${dist}
	data_root=${data_root}"_"${dist}
	subfolder="things_data_v2_"${dist}
      ;;
    --no_embed)
	embed=0
      ;;	
    --no_align)
	align=0
      ;;	
    --no_probe)
	probe=0
      ;;	
    *)
    echo "Invalid argument: " $1
  esac
shift
done

# Check if prefix matches one of the specified cases
case "$prefix" in
    ""|"conditional"|"textlast"|"textlastcapt"|"conditionalcapt"|"textlastcapt2"|"conditionalcapt2"|"optim"|"conditionaloptim"|"optimx1"|"conditionaloptimx1")
        echo "Valid prefix: '$prefix'"
        ;;
    *)
        echo "Invalid prefix: '$prefix'. Aborting."
        exit 1
        ;;
esac

if [ "$base_model" == "sd1" ]
then
	base_model="diffusion_runwayml/stable-diffusion-v1-5"
	base_modules=( "up_blocks.0.resnets.1" "up_blocks.1.resnets.1" "up_blocks.2.resnets.1" "up_blocks.3.resnets.1" "mid_block" "down_blocks.0.resnets.1" "down_blocks.1.resnets.1" "down_blocks.2.resnets.1" "down_blocks.3.resnets.1" )
	if [ "$prefix" == "conditionaloptim" ] || [ "$prefix" == "optim" ]
	then
		path_to_caption_dict="/home/llorenz/things_playground/things_data/optim_sd1.npy"
	elif [ "$prefix" == "conditionaloptimx1" ] || [ "$prefix" == "optimx1" ]
	then
		path_to_caption_dict="/home/llorenz/things_playground/things_data/optimx1_sd1.npy"
	fi
elif [ "$base_model" == "sd2" ]
then
	base_model="diffusion_stabilityai/stable-diffusion-2-1"
	base_modules=( "up_blocks.0.resnets.1" "up_blocks.1.resnets.1" "up_blocks.2.resnets.1" "up_blocks.3.resnets.1" "mid_block" "down_blocks.0.resnets.1" "down_blocks.1.resnets.1" "down_blocks.2.resnets.1" "down_blocks.3.resnets.1" )
	if [ "$prefix" == "conditionaloptim" ] || [ "$prefix" == "optim" ]
	then
		path_to_caption_dict="/home/llorenz/things_playground/things_data/optim_sd2.npy"
	elif [ "$prefix" == "conditionaloptimx1" ] || [ "$prefix" == "optimx1" ]
	then
		path_to_caption_dict="/home/llorenz/things_playground/things_data/optimx1_sd2.npy"
	fi
elif [ "$base_model" == "sd2t" ]
then
	base_model="diffusion_stabilityai/sd-turbo"
	base_modules=( "up_blocks.0.resnets.1" "up_blocks.1.resnets.1" "up_blocks.2.resnets.1" "up_blocks.3.resnets.1" "mid_block" "down_blocks.0.resnets.1" "down_blocks.1.resnets.1" "down_blocks.2.resnets.1" "down_blocks.3.resnets.1" )
	if [ "$prefix" == "conditionaloptim" ] || [ "$prefix" == "optim" ]
	then
		path_to_caption_dict="/home/llorenz/things_playground/things_data/optim_sd2t.npy"
	elif [ "$prefix" == "conditionaloptimx1" ] || [ "$prefix" == "optimx1" ]
	then
		path_to_caption_dict="/home/llorenz/things_playground/things_data/optimx1_sd2t.npy"
	fi
fi

if [ "$prefix" == "textlast" ] || [ "$prefix" == "textlastcapt" ] || [ "$prefix" == "textlastcapt2" ] || [ "$prefix" == "optim" ] || [ "$prefix" == "optimx1" ]
then
	base_modules=( "mid_block" )
	noise=(5)
else
	noise=(1 5 10 20 30 40 50 60 70 80 90)
	# noise=(15 25 30 35 45 55) # 70)
	# noise=(60 70 80 90) # 70)
	noise=(90) # 
fi

base_modules=( "mid_block" "up_blocks.0.resnets.1" "up_blocks.1.resnets.1" ) # "up_blocks.2.resnets.1" ) # "down_blocks.1.resnets.1" )
base_modules=( "down_blocks.0.resnets.1" )

base_models=()
for n in "${noise[@]}"; do
    base_models+=("${base_model}_${n}")
done


if [ "$prefix" == "textlastcapt" ] || [ "$prefix" == "conditionalcapt" ]
then
	path_to_caption_dict="/home/llorenz/things_playground/things_data/caption_dict.npy"
elif [ "$prefix" == "textlastcapt2" ] || [ "$prefix" == "conditionalcapt2" ]
then
	path_to_caption_dict="/home/llorenz/things_playground/things_data/captionsLavis.npy"
fi

# Check if prefix is not empty and append '-'
if [ -n "$prefix" ]; then
    prefix="${prefix}-"
fi

models=()
modules=()
sources=()

# Generate combinations
for item1 in "${base_models[@]}"; do
    for item2 in "${base_modules[@]}"; do
        models+=("$prefix""$item1")
        modules+=("$item2")
	sources+=("diffusion")
    done
done

############################### EMBEDDING
if [ $embed == 1 ]
then
	printf "\nStarted embedding for ${models[$SLURM_ARRAY_TASK_ID]} and ${modules[$SLURM_ARRAY_TASK_ID]} at $(date)\n"
	
	srun python3 main_embed.py \
	--path_to_caption_dict=${path_to_caption_dict} \
	--path_to_model_dict=${path_to_model_dict} \
	--data_root $data_root \
	--things_root $things_root \
	--model "${models[$SLURM_ARRAY_TASK_ID]}" \
	--module "${modules[$SLURM_ARRAY_TASK_ID]}" \
	--source ${sources[$SLURM_ARRAY_TASK_ID]} \
	${extra_embed}
	printf "\nFinished embedding for ${models[$SLURM_ARRAY_TASK_ID]} at $(date)\n"
else
	echo "Skipping embedding."
fi

############################### ALIGNING
if [ $align == 1 ]
then
	printf "\nStarted alignment eval for ${models[$SLURM_ARRAY_TASK_ID]} and ${modules[$SLURM_ARRAY_TASK_ID]} at $(date)\n"
	
	srun python3 main_align.py \
	--data_root $data_root \
	--things_root $things_root \
	--model "${models[$SLURM_ARRAY_TASK_ID]}" \
	--module "${modules[$SLURM_ARRAY_TASK_ID]}" \
	--source "${sources[$SLURM_ARRAY_TASK_ID]}" \
	${extra_align}
	printf "\nFinished alignment eval for ${models[$SLURM_ARRAY_TASK_ID]} at $(date)\n"
else
	echo "Skipping alignment eval."
fi

############################### PROBING
if [ $probe == 1 ]
then
	model_source="diffusion";
	lambdas=( "0.1" ); # ( "1.0" "0.1" "0.01" ); 
	logdir="./logs/${dataset}/probing/${models[$SLURM_ARRAY_TASK_ID]}/{modules[$SLURM_ARRAY_TASK_ID]}";
	mkdir -p $logdir;
	
	printf "\nStarted $probing for ${models[$SLURM_ARRAY_TASK_ID]} and ${modules[$SLURM_ARRAY_TASK_ID]} at $(date)\n"
	
	for lmbda in "${lambdas[@]}"; do
		srun python3 main_probing.py \
	--data_root $data_root \
	--dataset $dataset \
	--model "${models[$SLURM_ARRAY_TASK_ID]}" \
	--module "${modules[$SLURM_ARRAY_TASK_ID]}" \
	--source ${model_source} \
	--lmbda $lmbda \
	--use_bias \
	--probing_root $probing_root \
	--log_dir $log_dir \
	--device "gpu" \
	--num_processes 8 \
	--subfolder ${subfolder} \
	${extra_probe} 
	# >> ${logdir}/probing.out
	done
	printf "\nFinished $probing for ${models[$SLURM_ARRAY_TASK_ID]} for ${dataset} at $(date)\n"
else
	echo "Skipping probing."
fi