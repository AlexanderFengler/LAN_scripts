#!/bin/bash

# Run configs (parameters to script) ----------------
model='angle'

# Network training configs -----------
network_type='choicep'
#network_partition=gpu # batch / gpu
optimizer='adam'

# Number of networks to train (from config)
number_networks=36

# How many epochs to train the network for ?
network_n_epochs=20
# How many simulated trials per call to the simulator ?
data_gen_n_samples_per_sim=2000
# How many parameter sets do we request ? 
data_gen_n_parameter_sets=5000
# How many training examples do we harvest from a given parameter set
data_gen_n_training_examples_per_parameter_set=2000

network_training_config_file='/users/afengler/data/proj_ak_lan/akili/data/config_files/lan/network/'$network_type'/'$model'/train_config_opt_'$optimizer'_n_'$data_gen_n_samples_per_sim'_dt_0.001_nps_'$data_gen_n_parameter_sets'_npts_'$data_gen_n_training_examples_per_parameter_set'_architecture_search.pickle'

networks_path='/users/afengler/data/proj_ak_lan/akili/data/torch_models/lan/'$network_type'/'$model'/'

echo 'Config file passed to sbatch_network_training.sh'
echo $network_training_config_file
echo $networks_path

# Train networks ----
# sbatch -p gpu --gres=gpu:1  --account=carney-frankmj-condo  \
#                                       --array=0-$number_networks sbatch_network_training.sh \
#                                       --config_file $network_training_config_file \
#                                       --output_folder $networks_path

sbatch -p batch --account=carney-frankmj-condo  \
                --array=0-18 sbatch_network_training.sh \
                --config_file $network_training_config_file \
                --output_folder $networks_path
                
sbatch -p batch --array=19-36 sbatch_network_training.sh \
                --config_file $network_training_config_file \
                --output_folder $networks_path