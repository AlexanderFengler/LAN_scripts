#!/bin/bash

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
