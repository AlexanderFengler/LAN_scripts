#!/bin/bash

# CONFIGS ------
#accounts / carney-frankmj-condo / carney-brainstorm-condo  
source /users/afengler/.bashrc
conda deactivate
conda deactivate
conda activate lanfactory

# Run configs (parameters to script) ----------------
model='angle'

while [ ! $# -eq 0 ]
    do
        case "$1" in
             --model | -m)
                model=$2
        esac
        shift 2
    done

# Data generation configs --------

# How many simulated trials per call to the simulator?
data_gen_n_samples_per_sim=2000 
# How many parameter sets do we request? 
data_gen_n_parameter_sets=5000
# How many training examples do we harvest from a given parameter set?
data_gen_n_training_examples_per_parameter_set=2000 

# Where is the generator config file stored?
data_generation_config_file='/users/afengler/data/proj_ak_lan/akili/data/config_files/lan/data_generation/'$model'/'\
'nsim_'$data_gen_n_samples_per_sim'_dt_0.001_nps_'$data_gen_n_parameter_sets\
'_npts_'$data_gen_n_training_examples_per_parameter_set'.pickle'

echo $data_generation_config_file

sbatch -p batch --array=0-150 sbatch_data_generation.sh --config_file $data_generation_config_file

sbatch -p batch --account=carney-frankmj-condo --array=0-150 sbatch_data_generation.sh --config_file $data_generation_config_file