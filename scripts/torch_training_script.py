import psutil 
import argparse
import lanfactory
from copy import deepcopy
import os
import pickle
import torch
import random
import numpy as np

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
    print('Arguments passed: ', args)
        
    n_workers = min(12, psutil.cpu_count(logical = False) - 2)
    print('Number of workers we assign to the DataLoader: ', n_workers)

    # Load config dict
    if args.config_dict_key is None:
        config_dict = pickle.load(open(args.config_file, 'rb'))[0]
    else:
        config_dict = pickle.load(open(args.config_file, 'rb'))[args.config_dict_key]
    
    print('CONFIG DICT')
    print(config_dict)
    
    train_config = config_dict['train_config']
    network_config = config_dict['network_config']
    
    print('TRAIN CONFIG')
    print(train_config)
    
    print('NETWORK CONFIG')
    print(network_config)
    
    print('CONFIG DICT')
    print(config_dict)
    
    file_list = os.listdir(config_dict['training_data_folder'])
    valid_file_list = np.array([config_dict['training_data_folder'] + '/' + \
                         file_ for file_ in file_list if config_dict['training_file_identifier'] in file_])
    random.shuffle(valid_file_list)
    n_training_files = min(len(valid_file_list), train_config['n_training_files'])
    val_idx_cutoff = int(config_dict['train_val_split'] * n_training_files)
    
    print('NUMBER OF TRAINING FILES FOUND: ')
    print(len(valid_file_list))
          
    print('NUMBER OF TRAINING FILES USED: ')
    print(n_training_files)
          
    if torch.cuda.device_count() > 0:
          batch_size = train_config['gpu_batch_size']
    else:
          batch_size = train_config['cpu_batch_size']
    
    # Make the dataloaders
    my_train_dataset = lanfactory.trainers.DatasetTorch(file_IDs = valid_file_list[:val_idx_cutoff], #train_config['training_files'],
                                                        batch_size = batch_size,
                                                        label_prelog_cutoff_low = train_config['label_prelog_cutoff_low'])
    
    my_dataloader_train = torch.utils.data.DataLoader(my_train_dataset,
                                                      shuffle = train_config['shuffle_files'],
                                                      batch_size = None,
                                                      num_workers = n_workers,
                                                      pin_memory = True)
    
    my_val_dataset = lanfactory.trainers.DatasetTorch(file_IDs = valid_file_list[val_idx_cutoff:], #train_config['validation_files'],
                                                      batch_size = batch_size,
                                                      label_prelog_cutoff_low = train_config['label_prelog_cutoff_low'])
    
    my_dataloader_val = torch.utils.data.DataLoader(my_val_dataset,
                                                    shuffle = train_config['shuffle_files'],
                                                    batch_size = None,
                                                    num_workers = n_workers,
                                                    pin_memory = True)
    
    # Load network
    net = lanfactory.trainers.TorchMLP(network_config = deepcopy(network_config),
                                       input_shape = my_train_dataset.input_dim,
                                       save_folder = args.output_folder,
                                       generative_model_id = network_config['model_id'])

    # Save configs with model_id attached
    lanfactory.utils.save_configs(model_id = net.model_id + '_torch_',
                                  save_folder = args.output_folder + '/' + network_config['model_id'] + '/', 
                                  network_config = network_config, 
                                  train_config = train_config, 
                                  allow_abs_path_folder_generation = True)
    
    # Load model trainer
    my_model_trainer = lanfactory.trainers.ModelTrainerTorchMLP(train_config = deepcopy(train_config),
                                                                data_loader_train = my_dataloader_train,
                                                                data_loader_valid = my_dataloader_val,
                                                                model = net,
                                                                output_folder = args.output_folder, 
                                                                warm_start = False,
                                                                allow_abs_path_folder_generation = True)
    
    # Train model
    my_model_trainer.train_model(save_history = train_config['save_history'],
                                 save_model = True)