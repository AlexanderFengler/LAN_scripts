#!/bin/bash

# Run configs (parameters to script) ----------------
model="ddm"

# Network training configs -----------
network_type="cpn"
#network_partition=gpu # batch / gpu
optimizer="adam"
backend="jax"

# How many epochs to train the network for ?
#network_n_epochs=20
# How many simulated trials per call to the simulator ?
data_gen_n_samples_per_sim=20000
# How many parameter sets do we request ? 
data_gen_n_parameter_sets=1000
# How many training examples do we harvest from a given parameter set
data_gen_n_training_examples_per_parameter_set=2000

project_folder='/users/afengler/data/proj_lan_pipeline/LAN_scripts/'

network_training_config_path=$project_folder'/data/config_files/network/'$network_type'/'$model'/train_config_opt_'$optimizer'_n_'$data_gen_n_samples_per_sim'_dt_0.001_nps_'$data_gen_n_parameter_sets'_npts_'$data_gen_n_training_examples_per_parameter_set'_architecture_search.pickle'

networks_path=$project_folder'/local_tests/data/networks/'$network_type'/'$backend'/'$model'/'

echo 'Config file passed to sbatch_network_training.sh'
echo $network_training_config_file
echo $networks_path

# Train networks ----
if [ "$backend" == "jax" ]; then
    python -u scripts/jax_training_script.py --model $model \
                                             --config_path $network_training_config_path \
                                             --config_dict_key 0 \
                                             --network_folder $networks_path \
                                             --dl_workers 2
elif [ "$backend" == "torch" ]; then
    python -u scripts/torch_training_script.py --model $model \
                                             --config_path $network_training_config_path \
                                             --config_dict_key 0 \
                                             --network_folder $networks_path \
                                             --dl_workers 2
fi
