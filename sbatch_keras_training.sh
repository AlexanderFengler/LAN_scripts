#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J model_trainer

# priority
#SBATCH --account=carney-frankmj-condo

# output file
#SBATCH --output /users/afengler/batch_job_out/lanfactory_trainer_%A_%a.out

# Request runtime, memory, cores
#SBATCH --time=32:00:00
#SBATCH --mem=32G
#SBATCH -c 10
#SBATCH -N 1
#SBATCH -p gpu --gres=gpu:1
#SBATCH --array=0-89 # should be 89

# --------------------------------------------------------------------------------------

# Setup
source /users/afengler/.bashrc
module load cudnn/8.1.0
module load cuda/11.1.1
module load gcc/10.2

conda deactivate
conda deactivate
conda activate lanfactory

# Read in arguments:
config_dict_key=None
config_file=None
output_folder=/users/afengler/data/proj_lan_pipeline/LAN_scripts/data/models/
n_networks=1

while [ ! $# -eq 0 ]
    do
        case "$1" in
            --config_file | -c)
                config_file=$2
                ;;
            --output_folder | -o)
                output_folder=$2
                ;;
            --n_networks | -n)
                echo "passing number of networks $2"
                n_networks=$2
                ;;
        esac
        shift 2
    done

echo "The config file supplied is: $config_file"
echo "The config dictionary key supplied is: $config_dict_key"

x='teststr' # defined only for the check below (testing whether SLURM_ARRAY_TASK_ID is set)
if [ -z ${SLURM_ARRAY_TASK_ID} ];
then
    for ((i = 1; i <= $n_networks; i++))
        do
            echo "NOW TRAINING NETWORK: $i of $n_networks"
            echo "No array ID"
            python -u scripts/keras_training_script.py --config_file $config_file \
                                               --config_dict_key $config_dict_key \
                                               --output_folder $output_folder
        done
else
    for ((i = 1; i <= $n_networks; i++))
        do
            echo "NOW TRAINING NETWORK: $i of $n_networks "
            echo "Array ID $SLURM_ARRAY_TASK_ID "
            python -u scripts/keras_training_script.py --config_file $config_file \
                                               --config_dict_key $SLURM_ARRAY_TASK_ID \
                                               --output_folder $output_folder
        done
fi