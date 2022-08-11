#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J model_trainer

# priority
#SBATCH --account=carney-frankmj-condo

# output file
#SBATCH --output slurm/slurm_model_trainer_%A_%a.out

# Request runtime, memory, cores
#SBATCH --time=18:00:00
#SBATCH --mem=32G
#SBATCH -c 8
#SBATCH -N 1
##SBATCH -p gpu --gres=gpu:1
##SBATCH --array=0-8 # should be 89

# --------------------------------------------------------------------------------------

# Setup
source /users/afengler/.bashrc
module load cudnn/8.1.0
module load cuda/11.1.1
#module load cuda/11.3.1
module load gcc/10.2

conda deactivate
conda deactivate
conda activate lanfactory

# Read in arguments:
config_dict_key=None
config_file=None
output_folder=/users/afengler/data/proj_lan_pipeline/LAN_scripts/data/
n_networks=2
dl_backend=torch
#gpu=1

echo "arguments passed to sbatch_network_training.sh $#"

while [ ! $# -eq 0 ]
    do
        case "$1" in
            --config_file | -c)
                echo "passing config file $2"
                config_file=$2
                ;;
            --output_folder | -o)
                echo "passing output_folder $2"
                output_folder=$2
                ;;
            --n_networks | -n)
                echo "passing number of networks $2"
                n_networks=$2
                ;;
            --dl_backend | -n)
                echo "passing deep learning backend specification: $2"
                dl_backend=$2
        esac
        shift 2
    done

echo "The config file supplied is: $config_file"
echo "The config dictionary key supplied is: $config_dict_key"

if [[ $dl_backend == 'keras' ]];
then 
    output_folder="${output_folder}keras_models/"
elif [[ $dl_backend == 'torch' ]];
then 
    output_folder="${output_folder}torch_models/"
fi

echo "Output folder is: $output_folder"


x='teststr' # defined only for the check below (testing whether SLURM_ARRAY_TASK_ID is set)
if [ -z ${SLURM_ARRAY_TASK_ID} ];
then
    for ((i = 1; i <= $n_networks; i++))
        do
            echo "NOW TRAINING NETWORK: $i of $n_networks"
            echo "No array ID"
            if [[ $dl_backend == 'keras' ]];
            then
                python -u scripts/keras_training_script.py --config_file $config_file \
                                                           --config_dict_key $config_dict_key \
                                                           --output_folder $output_folder
            elif [[ $dl_backend == 'torch' ]];
            then
                python -u scripts/torch_training_script.py --config_file $config_file \
                                                           --config_dict_key $config_dict_key \
                                                           --output_folder $output_folder
            fi
        done
else
    for ((i = 1; i <= $n_networks; i++))
        do
            echo "NOW TRAINING NETWORK: $i of $n_networks "
            echo "Array ID $SLURM_ARRAY_TASK_ID "
            if [[ $dl_backend == 'keras' ]];
            then
                python -u scripts/keras_training_script.py --config_file $config_file \
                                                           --config_dict_key $SLURM_ARRAY_TASK_ID \
                                                           --output_folder $output_folder
            elif [[ $dl_backend == 'torch' ]];
            then
                python -u scripts/torch_training_script.py --config_file $config_file \
                                                           --config_dict_key $SLURM_ARRAY_TASK_ID \
                                                           --output_folder $output_folder
            fi
        done
fi