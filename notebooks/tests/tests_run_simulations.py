# MAKE CONFIGS
from ssms.basic_simulators import simulator
from ssms.config import data_generator_config
from ssms.config import model_config
from ssms.dataset_generators import data_generator
from copy import deepcopy

# Generator Config

# (We start from a supplied example in the ssms package)
ddm_generator_config = deepcopy(data_generator_config['lan']) 

# Specify generative model 
# (one from the list of included models in the ssms package / or a single string)
ddm_generator_config['dgp_list'] = 'ddm'

# Specify number of parameter sets to simulate
ddm_generator_config['n_parameter_sets'] = 1000

# Specify how many samples a simulation run should entail
# (To construct an empirical likelihood)
ddm_generator_config['n_samples'] = 10000

# Specify how many training examples to extract from 
# a single parameter vector
ddm_generator_config['n_training_samples_by_parameter_set'] = 2000

# Specify folder in which to save generated data
ddm_generator_config['output_folder'] = 'data/training_data/ddm_high_prec/'

# Model Config
ddm_model_config = model_config['ddm']

# MAKE DATA
n_datasets = 20

if __name__ == "__main__":

    # Instantiate a data generator (we pass our configs)
    my_dataset_generator = data_generator(generator_config = ddm_generator_config,
                                          model_config = ddm_model_config)

    for i in range(n_datasets):
        print('Dataset: ', i + 1, ' of ', n_datasets)
        training_data = my_dataset_generator.generate_data_training_uniform(save = True,
                                                                            verbose = False)