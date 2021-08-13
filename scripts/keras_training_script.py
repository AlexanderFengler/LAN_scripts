import argparse
import lanfactory
from copy import deepcopy
import os
import pickle

def none_or_str(value):
    print('none_or_str')
    print(value)
    print(type(value))
    if value == 'None':
        return None
    return value

def none_or_int(value):
    print('none_or_int')
    print(value)
    print(type(value))
    #print('printing type of value supplied to non_or_int ', type(int(value)))
    if value == 'None':
        return None
    return int(value)

if __name__ == "__main__":
    
    # Interface ----
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--config_file",
                     type = none_or_str,
                     default = None)
    CLI.add_argument('--config_dict_key',
                     type = none_or_int,
                     default = None)
    CLI.add_argument('--output_folder',
                     type = none_or_str,
                     default = None)
    
    args = CLI.parse_args()
    print(args)

    # Load config dict
    if args.config_dict_key is None:
        config_dict = pickle.load(open(args.config_file, 'rb'))
    else:
        config_dict = pickle.load(open(args.config_file, 'rb'))[args.config_dict_key]
        
    train_config = config_dict['train_config']
    network_config = config_dict['network_config']
    
    print('TRAIN CONFIG')
    print(train_config)
    
    
    
    print('NETWORK CONFIG')
    print(network_config)
    
    # Make data-generators
    my_generator_train = lanfactory.trainers.DataGenerator(file_IDs = train_config['training_files'],
                                                           batch_size = train_config['batch_size'],
                                                           shuffle = train_config['shuffle_files'],
                                                           label_prelog_cutoff_low = train_config['label_prelog_cutoff_low'])
    my_generator_val = lanfactory.trainers.DataGenerator(file_IDs = train_config['validation_files'],
                                                         batch_size = train_config['batch_size'],
                                                         shuffle = train_config['shuffle_files'],
                                                         label_prelog_cutoff_low = train_config['label_prelog_cutoff_low'])

    # Make the Keras Model
    my_keras_model = lanfactory.trainers.KerasModel(network_config = deepcopy(network_config),
                                                    input_shape = my_generator_train.input_dim,
                                                    save_folder = args.output_folder,
                                                    generative_model_id = network_config['model_id'])

    # Save configs with model_id attached
    lanfactory.utils.save_configs(model_id = my_keras_model.model_id,
                                  save_folder = args.output_folder, #'/users/afengler/data/proj_lan_pipeline/LAN_scripts/data/lan_mlp/models',
                                  network_config = network_config, #my_network_config,
                                  train_config = train_config, #my_train_config,
                                  allow_abs_path_folder_generation = True)

    my_model_trainer = lanfactory.trainers.ModelTrainerKerasSeq(train_config = deepcopy(train_config),
                                                                data_generator_train = my_generator_train,
                                                                data_generator_val = my_generator_val,
                                                                model = my_keras_model,
                                                                output_folder = args.output_folder, #'/users/afengler/data/proj_lan_pipeline/LAN_scripts/data/lan_mlp/models/',
                                                                warm_start = False,
                                                                allow_abs_path_folder_generation = True)

    my_model_trainer.train_model(save_history = train_config['save_history'])