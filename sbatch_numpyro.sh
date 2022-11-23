#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J numpyro_sampler

# priority
#SBATCH --account=carney-frankmj-condo

# output file
#SBATCH --output slurm/numpyro_sampler_%A_%a.out

# Request runtime, memory, cores
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH -c 8
#SBATCH -N 1
#SBATCH -p gpu --gres=gpu:1
#SBATCH --array=0-9

# --------------------------------------------------------------------------------------

# Setup
source /users/afengler/.bashrc
module load cudnn/8.1.0
module load cuda/11.1.1
#module load cuda/11.3.1
module load gcc/10.2

conda deactivate
conda deactivate
conda activate pymc-gpu

# Read in arguments:
model=ddm
modeltype=singlesubject
nwarmup=100
nmcmc=100
idrange=10

echo "arguments passed to sbatch_network_training.sh $#"

nvidia-smi

while [ ! $# -eq 0 ]
    do
        case "$1" in
            --model | -m)
                echo "passing config file $2"
                model=$2
                ;;
            --modeltype | -t)
                echo "passing output_folder $2"
                modeltype=$2
                ;;
            --nwarmup | -w)
                echo "passing number of networks $2"
                nwarump=$2
                ;;
            --nmcmc | -m)
                echo "passing number of networks $2"
                nmcmc=$2
                ;;
            --idrange | -i)
                echo "passing deep learning backend specification: $2"
                idrange=$2
        esac
        shift 2
    done

echo "Output folder is: $output_folder"

python -u run_inference_numpyro.py --model $model \
                                   --modeltype $modeltype \
                                   --nwarmup $nwarmup \
                                   --nmcmc $nmcmc \
                                   --idmin $((SLURM_ARRAY_TASK_ID*idrange)) \
                                   --idmax $((SLURM_ARRAY_TASK_ID*idrange + idrange))