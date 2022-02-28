#!/bin/bash

# Run Data Generation Script
# config_file: training config file generated with config_constructor.ipynb
# Oscar settings are in sbatch_data_generation.sh

# sbatch sbatch_data_generation.sh --config_file /users/afengler/data/proj_lan_pipeline/LAN_scripts/config_files/ddm_par2_no_bias_nsim_200000_dt_0.001_nps_5000_npts_2000.pickle

# Run Network Training Script
# config_file: network training config file generated with config_constructor.ipynb
# output_folder: where to store the training data ?
# n_networks: specifies how many networks are going to be trained for a given configuration 
# (useful to check if training results are relatively constant)
# dl_backed: could have been keras and torch --> only torch really relevant anymore (can keep argument at default)
# Oscar setting are in sbatch_network_generation.sh itself

sbatch sbatch_network_generation.sh --config_file /users/afengler/data/proj_lan_pipeline/LAN_scripts/config_files/gpu_torch_network_train_config_ddm_par2_no_bias_nsim_200000_dt_0.001_nps_5000_npts_2000_architecture_search.pickle
#                                     --output_folder \
#                                     --n_networks \
#                                     --dl_backend \