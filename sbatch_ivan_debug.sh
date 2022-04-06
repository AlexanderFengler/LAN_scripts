#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J ivan_debug

# priority
#SBATCH --account=carney-frankmj-condo

# output file
#SBATCH --output /users/afengler/batch_job_out/ivan_debug_%A_%a.out

# Request runtime, memory, cores
#SBATCH --time=18:00:00
#SBATCH --mem=32G
#SBATCH -c 10
#SBATCH -N 1
##SBATCH -p gpu --gres=gpu:1
#SBATCH --array=0-2 # should be 89

# --------------------------------------------------------------------------------------

# Setup
source /users/afengler/.bashrc
module load cudnn/8.1.0
module load cuda/11.1.1
module load gcc/10.2

conda deactivate
conda deactivate
conda activate lanfactory

python -u ivan_debug.py