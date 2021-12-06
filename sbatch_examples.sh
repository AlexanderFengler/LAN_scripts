#!/bin/bash

# Run Data Generation Script
# config_file: training config file generated with config_constructor.ipynb
# Oscar settings are in sbatch_data_generation.sh

sbatch sbatch_data_generation.sh --config_file 

# Run Network Training Script
# config_file: network training config file generated with config_constructor.ipynb
# output_folder: where to store the training data ?
# n_networks: specifies how many networks are going to be trained for a given configuration 
# (useful to check if training results are relatively constant)
# dl_backed: could have been keras and torch --> only torch really relevant anymore (can keep argument at default)
# Oscar setting are in sbatch_network_generation.sh itself

sbatch sbatch_network_generation.sh --config_file \
                                    --output_folder \
                                    --n_networks \
                                    --dl_backend \