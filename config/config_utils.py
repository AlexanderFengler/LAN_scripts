import lanfactory
import ssms
import numpy as np
import os
import pickle
from copy import deepcopy


def make_train_network_configs(training_data_folder = None,
                               training_file_identifier = None,
                               train_val_split = 0.9, 
                               save_folder = '',
                               network_arg_dict = None,
                               train_arg_dict = None,
                               save_name = None):
    
    train_config = deepcopy(lanfactory.config.train_config_mlp)
    network_config = deepcopy(lanfactory.config.network_config_mlp)
    
    for key, val in network_arg_dict.items():
        network_config[key] = val
        
    for key, val in train_arg_dict.items():
        train_config[key] = val
    
    config_dict = {'network_config': network_config,
                    'train_config': train_config}
    
    if save_name is not None:
        if len(save_folder) > 0:
            
            if save_folder[-1] == '/':
                pass
            else:
                save_folder = save_folder + '/'
        
        # Create save_folder if not already there
        lanfactory.utils.try_gen_folder(folder = save_folder, allow_abs_path_folder_generation = True)
             
        # Dump pickle file
        pickle.dump(config_dict, open(save_folder + save_name, 'wb'))
    return config_dict

def make_data_generator_configs(model = 'ddm',
                                generator_approach = 'lan',
                                generator_network_type = 'mlp',
                                data_generator_arg_dict = None,
                                model_config_arg_dict = None,
                                save_name = None,
                                save_folder = ''):
    
    model_config = deepcopy(ssms.config.model_config[model])
    data_config = deepcopy(ssms.config.data_generator_config[generator_approach])[generator_network_type]
    #print(data_config)
    data_config['dgp_list'] = model
    
    for key, val in data_generator_arg_dict.items():
        data_config[key] = val
        
    for key, val in model_config_arg_dict.items():
        model_config[key] = val

    config_dict = {'model_config': model_config, 'data_config': data_config}
    
    if save_name is not None:
        if len(save_folder) > 0:
            
            if save_folder[-1] == '/':
                pass
            else:
                save_folder = save_folder + '/'
        
        # Create save_folder if not already there
        lanfactory.utils.try_gen_folder(folder = save_folder, 
                                        allow_abs_path_folder_generation = True)
                
        # Dump pickle file
        pickle.dump(config_dict, open(save_folder + save_name, 'wb'))
    
    return config_dict
    
    
    
    






