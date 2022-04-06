#!/bin/bash

# CONFIGS ------

model='ddm'
data_gen_n_samples_per_sim=2000
data_gen_n_parameter_sets=100
data_gen_n_training_examples_per_parameter_set=2000

network_partition=batch
network_n_epochs=2

param_recov_n_mcmc=200
param_recov_n_burn=100
param_recov_n_chains=2
param_recov_n_subjects=10
param_recov_n_trials_per_subject=1000

data_generation_config_file='/users/afengler/data/proj_lan_pipeline/LAN_scripts/config_files/'$model\
'_nsim_'$data_gen_n_samples_per_sim'_dt_0.001_nps_'$data_gen_n_parameter_sets\
'_npts_'$data_gen_n_training_examples_per_parameter_set'.pickle'

network_training_config_file='/users/afengler/data/proj_lan_pipeline/LAN_scripts/config_files/'\
'torch_network_train_config_'$model'_nsim_'$data_gen_n_samples_per_sim'_dt_0.001_nps_'$data_gen_n_parameter_sets\
'_npts_'$data_gen_n_training_examples_per_parameter_set'_architecture_search.pickle'

networks_path='/users/afengler/data/proj_lan_pipeline/LAN_scripts/data/torch_models/'$model

param_recov_path='/users/afengler/data/proj_lan_pipeline/LAN_scripts/'\
'data/parameter_recovery/'$model\
'/subj_'$param_recov_n_subjects'_trials_'$param_recov_n_trials_per_subject'/'\

param_recov_config_file=$param_recov_path$model'_parameter_recovery_run_config.pickle'

# ---------------

# RUN THE FULL TRAINING PIPELINE -----

# #Make initial data and network configs ----
# python -u scripts/make_data_and_network_configs.py --model $model \
#                        --data_gen_n_samples_per_sim $data_gen_n_samples_per_sim \
#                        --data_gen_n_parameter_sets $data_gen_n_parameter_sets \
#                        --data_gen_n_training_examples_per_parameter_set $data_gen_n_training_examples_per_parameter_set \
#                        --network_n_epochs $network_n_epochs

# # Generate Training data ----
# jobID_1=$(sbatch -p batch --array=0-100 sbatch_data_generation.sh --config_file $data_generation_config_file | cut -f 4 -d' ')

# # Train networks ----
# jobID_2=$(sbatch --dependency=afterok:$jobID_1 --array=0-8 -p $network_partition sbatch_network_training.sh --config_file $network_training_config_file | cut -f 4 -d' ')

# # # Make parameter recovery configs ----
# jobID_3=$(sbatch --dependency=afterok:$jobID_2 -p batch sbatch_make_parameter_recovery_configs.sh --networks_path $networks_path --model $model --param_recov_n_mcmc $param_recov_n_mcmc --param_recov_n_burn $param_recov_n_burn --param_recov_n_chains $param_recov_n_chains --param_recov_n_subjects $param_recov_n_subjects --param_recov_n_trials_per_subject $param_recov_n_trials_per_subject | cut -f 4 -d' ')

# # Run parameter recovery ----
# jobID_4=$(sbatch --dependency=afterok:$jobID_3 -p batch --array=0-100 sbatch_parameter_recovery.sh --config_file $param_recov_config_file --parallel 1 --model_idx_low 0 --model_idx_high 5 --model_in_hddm 1 | cut -f 4 -d' ')


# # Post process parameter recovery data
# jobID_5=$(sbatch --dependency=afterok:$jobID_4 -p batch sbatch_preprocess_param_recov_data.sh --model $model --networks_path $networks_path --param_recov_path $param_recov_path | cut -f 4 -d' ') 
# -------------------------------


# INDIVIDUAL SCRIPTS
# -------------------------------

# # Make initial data and network configs
# python -u scripts/make_data_and_network_configs.py --model $model \
#                        --data_gen_n_samples_per_sim $data_gen_n_samples_per_sim \
#                        --data_gen_n_parameter_sets $data_gen_n_parameter_sets \
#                        --data_gen_n_training_examples_per_parameter_set $data_gen_n_training_examples_per_parameter_set \
#                        --network_n_epochs $network_n_epochs

# # Make training data only
# sbatch -p batch --array=0-100 sbatch_data_generation.sh --config_file $data_generation_config_file

# # Train networks only
# sbatch --array=0-8 -p $network_partition sbatch_network_training.sh --config_file $network_training_config_file

# # Parameter recovery configs only
# sbatch -p batch sbatch_make_parameter_recovery_configs.sh --networks_path $networks_path --model $model --param_recov_n_mcmc $param_recov_n_mcmc --param_recov_n_burn $param_recov_n_burn --param_recov_n_chains $param_recov_n_chains --param_recov_n_subjects $param_recov_n_subjects --param_recov_n_trials_per_subject $param_recov_n_trials_per_subject

# # Parameter recovery only
# sbatch -p batch --array=0-100 sbatch_parameter_recovery.sh --config_file $param_recov_config_file --parallel 1 --model_idx_low 0 --model_idx_high 5 --model_in_hddm 1

# Postprocess only -----
sbatch sbatch_postprocess_param_recov_data.sh --model $model --networks_path $networks_path --param_recov_path $param_recov_path
# --------------------------------