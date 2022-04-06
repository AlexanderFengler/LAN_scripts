# Append system path to include the config scripts
import sys
import os
import pickle
from copy import deepcopy
import argparse

print('importing lanfactory')
import lanfactory

print('importing ssms')
import ssms

print('importing hddm')
import hddm

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from config import *
import config
#from model_performance_utils import *
import matplotlib.pyplot as plt

import tensorflow
import torch

import pandas as pd

import sklearn 
from sklearn.linear_model import LinearRegression
from kabuki.analyze import gelman_rubin

def extract_model_id(x):
    idx = x['model_file'].find('_model_')
    return x['model_file'][(idx-32):idx]

def get_gelman_rubin_ok(model = None,
                        gelman_rubin_dict = None,
                        tolerance = 1.05):
    gl_ok_tmp = 1
    
    for param_tmp in hddm.model_config.model_config[model]['params']:
        if param_tmp + '_trans' in gelman_rubin_dict.keys():
            if gelman_rubin_dict[param_tmp + '_trans'] > 1.05:
                gl_ok_tmp = 0
        elif gelman_rubin_dict[param_tmp] > 1.05:
            gl_ok_tmp = 0
    
    return gl_ok_tmp

def ground_truth_from_parameter_dict(data = None,
                                     base_params_only = True,
                                     params = None,
                                     include_std = 0):
    data = deepcopy(data)
    param_dict_new = {}
    
    if base_params_only:
        # go row by row in data
        for i in range(data.shape[0]):
            # parameter dictionary
            parameter_dict_tmp = data.iloc[i, data.columns.get_loc('parameter_dict')]
            # cycle through parameters
            for param_tmp in params:
                if i == 0:
                    param_dict_new[param_tmp] = []
                    param_dict_new[param_tmp + '_posterior_mean'] = []
                
                # Append ground truth parameters
                param_dict_new[param_tmp].append(parameter_dict_tmp[param_tmp])
                # Append posterior mean
                param_dict_new[param_tmp + '_posterior_mean'].append(data.iloc[i]['traces'][param_tmp].mean())
    
    return pd.DataFrame.from_dict(param_dict_new)

if __name__ == "__main__":
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--model",
                     type = str,
                     default = '')
    CLI.add_argument("--networks_path",
                     type = str,
                     default = '')
    CLI.add_argument("--param_recov_path",
                     type = str,
                     default = '')
    CLI.add_argument("--gelman_rubin_tolerance",
                     type = float,
                     default = 1.05)
    args = CLI.parse_args()
    print('Arguments passed: ', args)
    
    # Collect parameter recovery files and turn into dataframe --------------------
    param_recov_files_ = [args.param_recov_path + file_ for file_ in os.listdir(args.param_recov_path)]
    db_files_ = [file_ for file_ in param_recov_files_ if '_db_' in file_] # sample databases
    df_files_ = [file_ for file_ in param_recov_files_ if '_df_' in file_] # metadata 
    model_files_ = [file_ for file_ in param_recov_files_ if '_model_' in file_] # hddm models

    metadata_list = [pickle.load(open(df_files_[i], 'rb')) for i in range(len(df_files_))]
    metadata_df = pd.concat(metadata_list).reset_index()
    metadata_df['model_id'] = metadata_df.apply(extract_model_id, axis = 1)
    # -------------------------------------------------------------------------------
    
    # Collect netowrk and training data ---------------------------------------------
    model_wise_pds = []
    pds = {}
    
    # Initial dictionary
    # with model_id as keys
    for file_ in os.listdir(args.networks_path):
        #if filter_ in file_ an not (filter_ in models_tmp):
        if 'training_history' in file_:
            pd_tmp = pd.read_csv(args.networks_path + '/' + file_)
            pd_tmp['model_id'] = file_[:file_.find('_')]
            pd_tmp['filename'] = file_
            pd_tmp['path'] = args.networks_path + '/' + file_
            pd_tmp['model_type'] = args.model
            pds[file_[:file_.find('_')]] = pd_tmp

    # Extend torch model wise dictionary 
    # to include 'n_hidden_layers', 'size_hidden_layers'
    for m_id in pds.keys():
        for file_ in os.listdir(args.networks_path):
            if m_id in file_ and 'network_config' in file_:
                network_config_tmp = pickle.load(open(args.networks_path + '/' + file_, 'rb'))
                pds[m_id]['n_hidden_layers'] = len(network_config_tmp['layer_sizes']) - 1
                pds[m_id]['size_hidden_layers'] = network_config_tmp['layer_sizes'][0]

    # Turn the model_id wise dictionary into dataframe
    training_dat = pd.concat([pds[m_id] for m_id in pds.keys()]).reset_index(drop = True)

    # Pick only the best models for a given configuration
    # of 'n_hidden_layers', 'size_hidden_layers' etc.
    best_models_pds = []
    for n_h_l in training_dat['n_hidden_layers'].unique():
        for s_h_l in training_dat['size_hidden_layers'].unique():
            training_dat_sub = training_dat.loc[(training_dat['n_hidden_layers'] == n_h_l) & \
                                                (training_dat['size_hidden_layers'] == s_h_l) & \
                                                (training_dat['epoch'] == training_dat['epoch'].max()), :]
            val_loss_min = training_dat_sub['val_loss'].min()
            best_models_pds.append(training_dat_sub.loc[training_dat_sub['val_loss'] == val_loss_min, :])

    # Turn best_models dictionary into dataframe across architectures
    best_models_dat = pd.concat(best_models_pds)

    # Append dataframe to list of model_wise dataframes 
    model_wise_pds.append(best_models_dat)
    full_dat = pd.concat(model_wise_pds).reset_index(drop = True).drop(['Unnamed: 0'], axis = 1)

    # Pick only the rows concerning final epochs
    training_dat = training_dat.loc[training_dat['epoch'] == training_dat['epoch'].max(), :]

    # Join hddm metadata with network data
    out = pd.merge(full_dat, metadata_df, on=['model_id'])
    # ------------------------------------------------------------------------------
    
    # Add a column that provides binarized gelman_rubin for given recov_ids --------
    gelman_rubin_list = {}
    out['gelman_rubin_ok'] = 1

    for model_id in out['model_id'].unique():
        for recov_idx in out['recov_idx'].unique():
            print('model_id: ', model_id)
            print('recov_id: ', recov_idx)
            if len(out.loc[(out.model_id == model_id) & \
                                (out.recov_idx == recov_idx), 'chain_idx'].unique()) > 1:
                models = []
                for chain_idx in out['chain_idx'].unique():
                    tmp_path = out.loc[(out.model_id == model_id) & \
                                            (out.recov_idx == recov_idx) & \
                                                   (out.chain_idx == chain_idx), 'model_file'].values[0]
                    models.append(hddm.load(tmp_path))

                out.loc[(out.model_id == model_id) & \
                            (out.recov_idx == recov_idx), 'gelman_rubin_ok'] = get_gelman_rubin_ok(model = args.model,
                                                                                                   gelman_rubin_dict = gelman_rubin(models),
                                                                                                   tolerance = args.gelman_rubin_tolerance)
            elif out.loc[(out.model_id == model_id) & \
                                (out.recov_idx == recov_idx), :].shape[0] > 0:

                out.loc[(out.model_id == model_id) & \
                                (out.recov_idx == recov_idx), 'gelman_rubin_ok'] = 0
                
    # --------------------------------------------------------------------------------
                
    # Get DataFrame that stores the recovery r-squared values per parameter for each model -----
    params = hddm.model_config.model_config[args.model]['params']
    recovery_dict = {}
    recovery_dict_gl_ok = {}
    param_dfs = []
    param_dfs_gl_ok = []

    out_gl_ok = out.loc[out.gelman_rubin_ok == 1, :]
    for model_id in out['model_id'].unique():

        if not ('model_id' in recovery_dict.keys()):
            recovery_dict['model_id'] = []
            recovery_dict_gl_ok['model_id'] = []

        recovery_dict['model_id'].append(model_id)
        recovery_dict_gl_ok['model_id'].append(model_id)
        data_tmp = out.loc[out.model_id == model_id]
        data_tmp_gl_ok = out_gl_ok.loc[out_gl_ok.model_id == model_id]

        param_df = ground_truth_from_parameter_dict(data = data_tmp,
                                                    params = params)
        param_df_gl_ok = ground_truth_from_parameter_dict(data = data_tmp_gl_ok,
                                                         params = params)

        param_df['model_id'] = model_id
        param_df_gl_ok['model_id'] = model_id

        param_dfs.append(param_df)
        param_dfs_gl_ok.append(param_df_gl_ok)

        for param_tmp in params:
            # Linear regression part:
            reg = LinearRegression().fit(np.expand_dims(param_df[param_tmp + '_posterior_mean'], 1), 
                                             param_df[param_tmp]) 
            reg_score_tmp = reg.score(np.expand_dims(param_df[param_tmp + '_posterior_mean'], 1), 
                                          param_df[param_tmp])

            reg_gl_ok = LinearRegression().fit(np.expand_dims(param_df_gl_ok[param_tmp + '_posterior_mean'], 1),
                                                   param_df_gl_ok[param_tmp])
            reg_score_tmp_gl_ok = reg.score(np.expand_dims(param_df_gl_ok[param_tmp + '_posterior_mean'], 1),
                                                param_df_gl_ok[param_tmp])

            if param_tmp + '_r2' not in recovery_dict.keys():
                print(param_tmp + '_r2')
                recovery_dict[param_tmp + '_r2'] = []
                recovery_dict_gl_ok[param_tmp + '_gl_ok_r2'] = []

            recovery_dict[param_tmp + '_r2'].append(reg_score_tmp)
            recovery_dict_gl_ok[param_tmp + '_gl_ok_r2'].append(reg_score_tmp_gl_ok)

    recovery_df = pd.DataFrame.from_dict(recovery_dict)
    recovery_df_gl_ok = pd.DataFrame.from_dict(recovery_dict_gl_ok)

    recovery_df['mean_r2'] = recovery_df.drop(columns = ['model_id']).mean(axis = 1).values
    recovery_df['mean_gl_ok_r2'] = recovery_df_gl_ok.drop(columns = ['model_id']).mean(axis = 1).values

    out = pd.merge(out, recovery_df, on=['model_id'])
    out = pd.merge(out, recovery_df_gl_ok, on=['model_id'])
    
    # ---------------------------------------------------------------------------------------------------
    
    # Save ----
    if not os.path.exists(args.param_recov_path + '/report'):
        os.mkdir(args.param_recov_path + '/report/')
    
    out.to_pickle(args.param_recov_path + '/report/' + args.model + '_recovery_main_dataframe.pickle')
    # ---------
    
    
    
    