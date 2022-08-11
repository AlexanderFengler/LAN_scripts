#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J param_recov_postprocess

# priority
#SBATCH --account=carney-frankmj-condo

# output file
#SBATCH --output slurm/slurm_param_recov_postprocess.out

# Request runtime, memory, cores
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH -N 1

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
model='ddm'
networks_path=None
param_recov_path=None
gelman_rubin_tolerance=1.05

while [ ! $# -eq 0 ]
    do
        case "$1" in
            --model | -m)
                model=$2
                ;;
            --networks_path | -l)
                networks_path=$2
                ;;
            --param_recov_path | -h)
                param_recov_path=$2
                ;;
            --gelman_rubin_tolerance | -g)
                gelman_rubin_tolerance=$2
        esac
        shift 2
    done

python -u scripts/postprocess_param_recov.py --model $model \
                                             --networks_path $networks_path \
                                             --param_recov_path $param_recov_path \
                                             --gelman_rubin_tolerance $gelman_rubin_tolerance 
