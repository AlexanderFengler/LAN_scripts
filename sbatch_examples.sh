#!/bin/bash

# Run Data Generation Script
# config_file: training config file generated with config_constructor.ipynb
# Oscar settings are in sbatch_data_generation.sh

# sbatch sbatch_data_generation.sh --config_file /users/afengler/data/proj_lan_pipeline/LAN_scripts/config_files/ddm_mic2_adj_gamma_conflict_no_bias_nsim_200000_dt_0.001_nps_5000_npts_2000.pickle

# Run Network Training Script
# config_file: network training config file generated swith config_constructor.ipynb
# n_networks: specifies how many networks are going to be trained for a given configuration 
# (useful to check if training results are relatively constant)

# Oscar setting are in sbatch_network_generation.sh itself

# sbatch sbatch_network_training.sh --config_file /users/afengler/data/proj_lan_pipeline/LAN_scripts/config_files/gpu_torch_network_train_config_mic2_adj_no_bias_nsim_200000_dt_0.001_nps_5000_npts_2000_architecture_search.pickle
#                                     #--n_networks \

# Run parameter recovery
base_folder='/users/afengler/data/proj_lan_pipeline/LAN_scripts/data/parameter_recovery'
model='angle'
data_signature='subj_10_trials_1000'

sbatch sbatch_parameter_recovery.sh --config_file $base_folder'/'$model'/'$data_signature'/'$model'_parameter_recovery_run_config.pickle' \
                                    --parallel 1 \
                                    --model_idx_low 0 \
                                    --model_idx_high 5 \
                                    --verbose 0 \
                                    --model_in_hddm 1