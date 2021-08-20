#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J data_generator

# priority
#SBATCH --account=carney-frankmj-condo

# output file
#SBATCH --output /users/afengler/batch_job_out/data_generator_%A_%a.out

# Request runtime, memory, cores
#SBATCH --time=8:00:00
#SBATCH --mem=16G
#SBATCH -c 12
#SBATCH -N 1
##SBATCH --array=1-300  # DO THIS FOR TRAINING DATA GENERATION
#SBATCH --array=1-500

# --------------------------------------------------------------------------------------

# BASIC SETUP
source /users/afengler/.bashrc
conda deactivate
conda deactivate
conda activate lanfactory

# Read in arguments:
# AF-TODO: Add new argument --use_array_id
config_dict_key=None
config_file=None

while [ ! $# -eq 0 ]
    do
        case "$1" in
            --config_file | -cf)
                config_file=$2
                ;;
            --config_dict_key | -cd)
                config_dict_key=$2
                ;;
        esac
        shift 2
    done

echo "The config file supplied is: $config_file"
echo "The config dictionary key supplied is: $config_dict_key"


#x='teststr' # defined only for the check below (testing whether SLURM_ARRAY_TASK_ID is set)
#if [ -z ${SLURM_ARRAY_TASK_ID} ];
#then
python -u scripts/data_generation_script.py --config_file $config_file #\
                                                #--config_dict_key $config_dict_key
#else
#python -u scripts/data_generation_script.py --config_file $config_file #\
                                                #--config_dict_key $SLURM_ARRAY_TASK_ID
#fi

# CONFIG DICT KEY DEPENDS ON ARRAY ID
# python -u data_generation_script.py --config_file $config_file \
#                                     --config_dict_key $SLURM_ARRAY_TASK_ID