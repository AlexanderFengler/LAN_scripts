#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J param_recov_config

# priority
#SBATCH --account=carney-frankmj-condo

# output file
#SBATCH --output slurm/slurm_param_recov_config_%A_%a.out

# Request runtime, memory, cores
#SBATCH --time=18:00:00
#SBATCH --mem=32G
#SBATCH -c 8
#SBATCH -N 1
##SBATCH -p gpu --gres=gpu:1
##SBATCH --array=0-10 # should be 89

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
param_recov_n_data_sets=1000
param_recov_n_subjects=10
param_recov_n_trials_per_subject=1000
param_recov_n_lans_to_pick=10
param_recov_n_burn=1000
param_recov_n_mcmc=5000
param_recov_n_chains=2

# Take care of input arguments
while [ ! $# -eq 0 ]
    do
        case "$1" in
            --networks_path | -n)
                networks_path=$2
                ;;
            --model | -m)
                model=$2
                ;;
            --param_recov_n_data_sets | -d)
                echo "passing number of networks $2"
                param_recov_n_data_sets=$2
                ;;
            --param_recov_n_subjects | -s)
                echo "passing deep learning backend specification: $2"
                param_recov_n_subjects=$2
                ;;
            --param_recov_n_trials_per_subject | -p)
                echo "passing number of networks $2"
                param_recov_n_trials_per_subject=$2
                ;;
            --param_recov_n_lans_to_pick| -l)
                echo "passing number of networks $2"
                param_recov_n_lans_to_pick=$2
                ;;
            --param_recov_n_burn | -b)
                echo "passing number of networks $2"
                param_recov_n_burn=$2
                ;;
            --param_recov_n_mcmc | -d)
                echo "passing number of networks $2"
                param_recov_n_mcmc=$2
                ;;
            --param_recov_n_chains | -d)
                echo "passing number of networks $2"
                pram_recov_n_chains=$2
        esac
        shift 2
    done
    
python -u scripts/make_param_recov_configs.py --model $model \
                        --networks_path $networks_path \
                        --param_recov_n_data_sets $param_recov_n_data_sets \
                        --param_recov_n_subjects $param_recov_n_subjects \
                        --param_recov_n_trials_per_subject $param_recov_n_trials_per_subject \
                        --param_recov_n_lans_to_pick $param_recov_n_lans_to_pick \
                        --param_recov_n_burn $param_recov_n_burn \
                        --param_recov_n_mcmc $param_recov_n_mcmc \
                        --param_recov_n_chains $param_recov_n_chains
