#!/bin/bash

# job name:
#SBATCH -J training_pipeline

# priority
##SBATCH --account=carney-frankmj-condo

# output file
#SBATCH --output slurm/slurm_training_pipeline.out

# Request runtime, memory, cores
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH -N 1
# ---------------------------------------------------------

# CONFIGS ------
#accounts / carney-frankmj-condo / carney-brainstorm-condo  
source /users/afengler/.bashrc
conda deactivate
conda deactivate
conda activate lanfactory

echo $SLURM_JOB_ACCOUNT

# Run configs (parameters to script) ----------------
config_generation=0
data_generation=0
network_training=0
parameter_recovery=0
parameter_recovery_config=0
parameter_recovery_postprocess=0
model='race_no_bias_3'

while [ ! $# -eq 0 ]
    do
        case "$1" in
             --model | -m)
                model=$2
                ;;
            --config_generation | -c)
                config_generation=$2
                ;;
            --data_generation | -d)
                data_generation=$2
                ;;
            --network_training | -n)
                network_training=$2
                ;;
            --parameter_recovery | -p)
                parameter_recovery=$2
                ;;
            --parameter_recovery_config | -l)
                parameter_recovery_config=$2
                ;;
            --parameter_recovery_postprocess | -r)
                parameter_recovery_postprocess=$2
        esac
        shift 2
    done

# Data generation configs --------
data_gen_n_samples_per_sim=200000 #200000  # How many simulated trials per call to the simulator ?
data_gen_n_parameter_sets=5000   #5000      # How many parameter sets do we request ? 
data_gen_n_training_examples_per_parameter_set=2000 # How many training examples do we harvest from a given parameter set

data_generation_config_file='/users/afengler/data/proj_lan_pipeline/LAN_scripts/config_files/'$model\
'_nsim_'$data_gen_n_samples_per_sim'_dt_0.001_nps_'$data_gen_n_parameter_sets\
'_npts_'$data_gen_n_training_examples_per_parameter_set'.pickle'

# Network training configs -----------
network_partition=gpu # batch / gpu
network_n_epochs=20 # how many epochs to train the network for ?

network_training_config_file='/users/afengler/data/proj_lan_pipeline/LAN_scripts/config_files/'\
'torch_network_train_config_'$model'_nsim_'$data_gen_n_samples_per_sim'_dt_0.001_nps_'$data_gen_n_parameter_sets\
'_npts_'$data_gen_n_training_examples_per_parameter_set'_architecture_search.pickle'

networks_path='/users/afengler/data/proj_lan_pipeline/LAN_scripts/data/torch_models/'$model

# Parameter recovery configs -----------
param_recov_n_mcmc=2000 # how many mcmc samples for a given inference run
param_recov_n_burn=1000 # how many burn in sampler for a given inference run
param_recov_n_chains=2 # how many chains ?
param_recov_n_subjects=10 # How many subjects ? (For hierarchical models > 1)
param_recov_n_trials_per_subject=1000 # How many trials per subject ?

param_recov_path='/users/afengler/data/proj_lan_pipeline/LAN_scripts/'\
'data/parameter_recovery/'$model\
'/subj_'$param_recov_n_subjects'_trials_'$param_recov_n_trials_per_subject'/'\

param_recov_config_file=$param_recov_path$model'_parameter_recovery_run_config.pickle'
# --------------------------------------

# INDIVIDUAL SCRIPTS

# Make initial data and network configs -------
if [[ $config_generation == 1 ]];
then
    python -u scripts/make_data_and_network_configs.py --model $model \
                                                       --data_gen_n_samples_per_sim $data_gen_n_samples_per_sim \
                                                       --data_gen_n_parameter_sets $data_gen_n_parameter_sets \
                                                       --data_gen_n_training_examples_per_parameter_set $data_gen_n_training_examples_per_parameter_set \
                                                       --network_n_epochs $network_n_epochs
fi
# --------------------------------------------

# Make training data only --------------------
if [[ $data_generation == 1]];
then
    # WITH ACCOUNT
    sbatch -p batch --account=$SLURM_JOB_ACCOUNT --array=0-100 sbatch_data_generation.sh --config_file $data_generation_config_file
    # WITHOUT ACCOUNT
    sbatch -p batch --array=0-200 sbatch_data_generation.sh --config_file $data_generation_config_file
fi
# -------------------------------------------

# Train networks ----------------------------
if [[ $network_training == 1 ]];
then
    sbatch --array=0-2 -p $network_partition --gres=gpu:1 sbatch_network_training.sh --config_file $network_training_config_file
fi
# -------------------------------------------

# Parameter recovery configs ----------------
if [[ $parameter_recovery_config ]];
then
    sbatch -p batch sbatch_make_parameter_recovery_configs.sh --networks_path $networks_path --model $model --param_recov_n_mcmc $param_recov_n_mcmc --param_recov_n_burn $param_recov_n_burn --param_recov_n_chains $param_recov_n_chains --param_recov_n_subjects $param_recov_n_subjects --param_recov_n_trials_per_subject $param_recov_n_trials_per_subject
fi
# -------------------------------------------

# Parameter recovery ------------------------
if [[ $parameter_recovery ]];
then
    sbatch -p batch --array=0-100 sbatch_parameter_recovery.sh --config_file $param_recov_config_file --parallel 1 --model_idx_low 0 --model_idx_high 5 --model_in_hddm 1
fi
# -------------------------------------------

# Postprocess  ------------------------------
if [[ $parameter_recovery_postprocess ]];
then
    sbatch sbatch_postprocess_param_recov_data.sh --model $model --networks_path $networks_path --param_recov_path $param_recov_path
fi
# -------------------------------------------



# EXAMPLE ON HOW TO USE DEPENDENT JOBS TO RUN THE WHOLE PIPELINE AT ONCE ----

# RUN THE FULL TRAINING PIPELINE -----
#Make initial data and network configs ----
# python -u scripts/make_data_and_network_configs.py --model $model \
#                        --data_gen_n_samples_per_sim $data_gen_n_samples_per_sim \
#                        --data_gen_n_parameter_sets $data_gen_n_parameter_sets \
#                        --data_gen_n_training_examples_per_parameter_set $data_gen_n_training_examples_per_parameter_set \
#                        --network_n_epochs $network_n_epochs

# # Generate Training data ----
# jobID_1=$(sbatch -p batch --account=$SLURM_JOB_ACCOUNT --array=0-100 sbatch_data_generation.sh --config_file $data_generation_config_file | cut -f 4 -d' ')

# # jobID_2=$(sbatch -p batch --account=carney-frankmj-condo --array=0-100 sbatch_data_generation.sh --config_file $data_generation_config_file | cut -f 4 -d' ')

# # Train networks ----
# jobID_2=$(sbatch --dependency=afterok:$jobID_1 --account=$SLURM_JOB_ACCOUNT --array=0-8 -p $network_partition sbatch_network_training.sh --gres=gpu:1 --config_file $network_training_config_file | cut -f 4 -d' ')

# # Make parameter recovery configs ----
# jobID_3=$(sbatch --dependency=afterok:$jobID_2 --account=$SLURM_JOB_ACCOUNT -p batch sbatch_make_parameter_recovery_configs.sh --networks_path $networks_path --model $model --param_recov_n_mcmc $param_recov_n_mcmc --param_recov_n_burn $param_recov_n_burn --param_recov_n_chains $param_recov_n_chains --param_recov_n_subjects $param_recov_n_subjects --param_recov_n_trials_per_subject $param_recov_n_trials_per_subject | cut -f 4 -d' ')

# # Run parameter recovery ----
# jobID_4=$(sbatch --dependency=afterok:$jobID_3 --account=$SLURM_JOB_ACCOUNT -p batch --array=0-100 sbatch_parameter_recovery.sh --config_file $param_recov_config_file --parallel 1 --model_idx_low 0 --model_idx_high 5 --model_in_hddm 1 | cut -f 4 -d' ')


# # Post process parameter recovery data
# jobID_5=$(sbatch --dependency=afterok:$jobID_4 --account=$SLURM_JOB_ACCOUNT -p batch sbatch_preprocess_param_recov_data.sh --model $model --networks_path $networks_path --param_recov_path $param_recov_path | cut -f 4 -d' ') 
# # # -------------------------------