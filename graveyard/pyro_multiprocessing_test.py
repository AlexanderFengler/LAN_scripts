import torch
import pyro
import pyro.distributions as dist
import argparse
import matplotlib
import matplotlib.pyplot as plt
from torch.distributions.constraints import positive

import logging
import os

#import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import scipy

import pyro
import ssms
import lanfactory
#torch.set_default_dtype(torch.float64)
torch.set_default_dtype(torch.float32)

from lanfactory.trainers.torch_mlp import TorchMLP 

import lanfactory
import ssms
from copy import deepcopy

smoke_test = ('CI' in os.environ)
assert pyro.__version__.startswith('1.8.1')

pyro.enable_validation(True)
pyro.set_rng_seed(9)
logging.basicConfig(format='%(message)s', level=logging.INFO)

import math
from numbers import Real
from numbers import Number

import torch
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import _standard_normal, broadcast_all

from time import time

def sim_wrap(theta = torch.zeros(0), model = 'ddm'):
    theta = theta.numpy().astype(np.float32)
    out = ssms.basic_simulators.simulator(theta = theta,
                                          model = model,
                                          n_samples = 1,
                                          delta_t = 0.001,
                                          max_t = 20.0,
                                          no_noise = False,
                                          bin_dim = None,
                                          bin_pointwise = False)
    
    return torch.tensor(np.hstack([out['rts'].astype(np.float32), out['choices'].astype(np.float32)]))

class LoadTorchMLP:
    def __init__(self, 
                 model_file_path = None,
                 network_config = None,
                 input_dim = None):
        
        ##torch.backends.cudnn.benchmark = True
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model_file_path = model_file_path
        self.network_config = network_config
        self.input_dim = input_dim
        
        self.net = lanfactory.trainers.torch_mlp.TorchMLP(network_config = self.network_config,
                                                          input_shape = self.input_dim,
                                                          generative_model_id = None)
        self.net.load_state_dict(torch.load(self.model_file_path))
        self.net.to(self.dev)

    # AF-TODO: Seemingly LoadTorchMLPInfer is still not callable !
    @torch.no_grad()
    def __call__(self, x):
        return self.net(x)

    @torch.no_grad()
    def predict_on_batch(self, x = None):
        return self.net(torch.from_numpy(x).to(self.dev)).cpu().numpy()
    
    
class CustomTorchMLP:
    def __init__(self, state_dict, network_config):
        self.weights = []
        self.biases = []
        self.activations = deepcopy(network_config['activations'])
        self.net_depth = len(self.activations)
        self.state_dict = state_dict
        cnt = 0
        for obj in self.state_dict:
            if 'weight' in obj:
                self.weights.append(deepcopy(self.state_dict[obj]).T)
            elif 'bias' in obj:
                self.biases.append(torch.unsqueeze(deepcopy(self.state_dict[obj]), 0))
                
    def forward(self, input_tensor):
        tmp = input_tensor
        for i in range(0, self.net_depth - 1, 1):
            tmp = torch.tanh(torch.add(torch.matmul(tmp, self.weights[i]), self.biases[i]))
        tmp = torch.add(torch.matmul(tmp, self.weights[self.net_depth - 1]), self.biases[self.net_depth - 1])
        return tmp
    

class CustomTorchMLPMod(torch.nn.Module):
    def __init__(self, state_dict, network_config):
        super(CustomTorchMLPMod, self).__init__()
        self.weights = []
        self.biases = []
        self.activations = deepcopy(network_config['activations'])
        self.net_depth = len(self.activations)
        self.state_dict = state_dict
        cnt = 0
        for obj in self.state_dict:
            if 'weight' in obj:
                self.weights.append(deepcopy(self.state_dict[obj]).T)
            elif 'bias' in obj:
                self.biases.append(torch.unsqueeze(deepcopy(self.state_dict[obj]), 0))
        #super().__init
        
    def forward(self, input_tensor):
        tmp = input_tensor
        for i in range(0, self.net_depth - 1, 1):
            tmp = torch.tanh(torch.add(torch.matmul(tmp, self.weights[i]), self.biases[i]))
        tmp = torch.add(torch.matmul(tmp, self.weights[self.net_depth - 1]), self.biases[self.net_depth - 1])
        return tmp
    
class MyDDMh(dist.TorchDistribution):
#     arg_constraints = {'loc': constraints.interval(-1, 1),
#                        'scale': constraints.interval(0.0001, 10)
#                       }
    def __init__(self, v, a, z, t):
        self.net = network
        self.n_samples = n_samples_by_subject
        self.boundaries = model_config['param_bounds']
        self.out_of_bounds_val = -66.1
        self.v = v
        self.a = a
        self.z = z
        self.t = t
        
        if isinstance(v, Number): # and isinstance(a, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.v.size()
            
        super().__init__(batch_shape = batch_shape, event_shape = torch.Size((2,))) #torch.Size((2,))) # event_shape = (1,))
        
    def sample(self):
        theta = torch.vstack([self.v, self.a, self.z, self.t]).T
        return sim_wrap(theta = theta, model = 'ddm')
    
    def log_prob(self, value):
        
        if self.v.dim() == 3:
            dat_tmp = value.repeat((self.v.size()[0], 1, 1, 1))

            tmp_params = torch.stack([self.v, self.a, self.z, self.t], 
                                     dim = -1).tile((1, self.n_samples, 1, 1))
 
            net_in = torch.cat([tmp_params, dat_tmp], dim = -1)
            logp = torch.clip(self.net(net_in), min = -16.11)
            logp_squeezed = torch.squeeze(logp, dim = -1)
            
            # v constraint
            logp_squeezed = torch.where(net_in[:, :, :, 0] < torch.tensor(3.),
                                        logp_squeezed, 
                                        torch.tensor(self.out_of_bounds_val))
            logp_squeezed = torch.where(net_in[:, :, :, 0] > torch.tensor(-3.),
                                        logp_squeezed, 
                                        torch.tensor(self.out_of_bounds_val))

            # a constraint
            logp_squeezed = torch.where(net_in[:, :, :, 1] < torch.tensor(2.5),
                                        logp_squeezed, 
                                        torch.tensor(self.out_of_bounds_val))
            logp_squeezed = torch.where(net_in[:, :, :, 1] > torch.tensor(0.3), 
                                        logp_squeezed, 
                                        torch.tensor(self.out_of_bounds_val))

            # z constraint
            logp_squeezed = torch.where(net_in[:, :, :, 2] < torch.tensor(0.9),
                                        logp_squeezed, 
                                        torch.tensor(self.out_of_bounds_val))
            logp_squeezed = torch.where(net_in[:, :, :, 2] > torch.tensor(0.1),
                                        logp_squeezed, 
                                        torch.tensor(self.out_of_bounds_val))

            # t constraint
            logp_squeezed = torch.where(net_in[:, :, :, 3] < torch.tensor(2.0), 
                                        logp_squeezed, 
                                        torch.tensor(self.out_of_bounds_val))
            logp_squeezed = torch.where(net_in[:, :, :, 3] > torch.tensor(0.0),
                                        logp_squeezed, 
                                        torch.tensor(self.out_of_bounds_val))
            
            logp_squeezed = logp_squeezed #.unsqueeze(1)
        
        else: # single particle:
            tmp_params = torch.stack([self.v, self.a, self.z, self.t], 
                                     dim = -1).tile((self.n_samples, 1, 1))

            net_in = torch.cat([tmp_params, value], dim = -1)
            logp = torch.clip(self.net(net_in), min = -16.11)
            logp_squeezed = torch.squeeze(logp, dim = -1)

            # v constraint
            logp_squeezed = torch.where(net_in[:, :, 0] < torch.tensor(3.),
                                        logp_squeezed, 
                                        torch.tensor(self.out_of_bounds_val))
            logp_squeezed = torch.where(net_in[:, :, 0] > torch.tensor(-3.), 
                                        logp_squeezed, 
                                        torch.tensor(self.out_of_bounds_val))

            # a constraint
            logp_squeezed = torch.where(net_in[:, :, 1] < torch.tensor(2.5),
                                        logp_squeezed, 
                                        torch.tensor(self.out_of_bounds_val))
            logp_squeezed = torch.where(net_in[:, :, 1] > torch.tensor(0.3), 
                                        logp_squeezed, 
                                        torch.tensor(self.out_of_bounds_val))

            # z constraint
            logp_squeezed = torch.where(net_in[:, :, 2] < torch.tensor(0.9),
                                        logp_squeezed,
                                        torch.tensor(self.out_of_bounds_val))
            logp_squeezed = torch.where(net_in[:, :, 2] > torch.tensor(0.1), 
                                        logp_squeezed,
                                        torch.tensor(self.out_of_bounds_val))

            # t constraint
            logp_squeezed = torch.where(net_in[:, :, 3] < torch.tensor(2.0),
                                        logp_squeezed, 
                                        torch.tensor(self.out_of_bounds_val))
            logp_squeezed = torch.where(net_in[:, :, 3] > torch.tensor(0.0), 
                                        logp_squeezed, 
                                        torch.tensor(self.out_of_bounds_val))

        return logp_squeezed
    
def hierarchical_ddm_model(num_subjects, num_trials, data):
    #v_mu_mu = pyro.sample("v_mu_mu", dist.Uniform(-3, 3))
    v_mu_mu = pyro.sample("v_mu_mu", dist.Normal(0, 0.5))
    v_mu_std = pyro.sample("v_mu_std", dist.HalfNormal(100.))
    
    #a_mu_mu = pyro.sample("a_mu_mu", dist.Uniform(0.3, 2.5))
    a_mu_std = pyro.sample("a_mu_std", dist.HalfNormal(100.))
    a_mu_mu = pyro.sample("a_mu_mu", dist.Normal(1.5, 0.5))

    #z_mu_mu = pyro.sample("z_mu_mu", dist.Uniform(0.1, 0.9))
    z_mu_std = pyro.sample("z_mu_std", dist.HalfNormal(100.))
    z_mu_mu = pyro.sample("z_mu_mu", dist.Normal(0.5, 0.1))
    
    #t_mu_mu = pyro.sample("t_mu_mu", dist.Uniform(0.0, 2.0))
    t_mu_std = pyro.sample("t_mu_std", dist.HalfNormal(100.))
    t_mu_mu = pyro.sample("t_mu_mu", dist.Normal(1.0, 0.5))

    with pyro.plate("subjects", num_subjects) as subjects_plate:
        v_subj = pyro.sample("v_subj", dist.Normal(v_mu_mu, v_mu_std))
        a_subj = pyro.sample("a_subj", dist.Normal(a_mu_mu, a_mu_std))
        z_subj = pyro.sample("z_subj", dist.Normal(z_mu_mu, z_mu_std))
        t_subj = pyro.sample("t_subj", dist.Normal(t_mu_mu, t_mu_std))
        with pyro.plate("data", num_trials) as data_plate:
            return pyro.sample("obs", 
                               MyDDMh(v_subj, a_subj, z_subj, t_subj), 
                               obs = data) 
    
if __name__ == "__main__":

#     # Interface ----
#     CLI = argparse.ArgumentParser()
#     CLI.add_argument("--config_file",
#                      type = none_or_str,
#                      default = None)
#     CLI.add_argument('--config_dict_key',
#                      type = none_or_int,
#                      default = None)
    
#     args = CLI.parse_args()
#     print(args)
        
 
    # # Load torch net ----------------
    # Model
    
    model = "ddm" # for now only DDM (once we have choice probability models --> all models applicable)
    model_config = ssms.config.model_config[model].copy() # convenience
    
    
    network_config = pickle.load(open('nets/d27193a4153011ecb76ca0423f39a3e6_' + \
                                      'ddm_torch__network_config.pickle', 'rb'))

    print(network_config)

    # Initialize network class
    torch_net = TorchMLP(network_config = network_config,
                         input_shape = len(model_config['params']) + 2,
                         generative_model_id = None)

    # Load weights and biases
    torch_net.load_state_dict(torch.load('nets/d27193a4153011ecb76ca0423f39a3e6_' + \
                                         'ddm_torch_state_dict.pt', 
                              map_location=torch.device('cpu')))

    # Turn torch network usable for us
    custom_torch_net = CustomTorchMLPMod(torch_net.state_dict(), 
                                         network_config)
    custom_torch_net.eval()
    
    
    # Generate Data
    base_dim = 1
    n_subjects = 20
    n_samples_by_subject = 500
    buffer_coefficient = 0.5

    param_dict = {}
    data_list = []
    for param in model_config['params']:
        param_idx = model_config['params'].index(param)
        print('param')
        print(param)
        min_ = model_config['param_bounds'][0][param_idx]
        max_ = model_config['param_bounds'][1][param_idx]
        range_ = max_ - min_
        mean_ = (max_ + min_) / 2
        min_adj = mean_ - (0.5 * buffer_coefficient) * range_
        print(min_adj)
        max_adj = mean_ + (0.5 * buffer_coefficient) * range_
        print(max_adj)

        param_mu_mu = np.random.uniform(low = min_adj, high = max_adj) # potentially fix
        param_mu_std = np.random.uniform(low = 0.05, high = 0.1) # potentially fix
        param_mu = scipy.stats.norm.rvs(loc = param_mu_mu, scale = param_mu_std)
        param_std = scipy.stats.halfnorm.rvs(loc = 0, scale = 0.1) # potentially fix
        # param_std_std = scipy.stats.halfnorm(loc = 0, scale = 0.25) # potentially fix

        params_subj = np.random.normal(loc = param_mu, 
                                       scale = param_std, 
                                       size = n_subjects)

        param_dict[param + '_mu'] = param_mu.astype(np.float32)
        param_dict[param + '_std'] =  param_std.astype(np.float32)
        param_dict[param + '_subj'] = params_subj.astype(np.float32)

    print(param_dict)

    for i in range(n_subjects):
        v = torch.zeros(base_dim) + param_dict['v_subj'][i]
        a = torch.zeros(base_dim) + param_dict['a_subj'][i]
        z = torch.zeros(base_dim) + param_dict['z_subj'][i]
        t = torch.zeros(base_dim) + param_dict['t_subj'][i]

        theta = torch.vstack([v, a, z, t]).T
        theta = theta.tile((n_samples_by_subject, 1))
        out = sim_wrap(theta = theta)
        # theta = torch.hstack([theta, out])
        data_list.append(out)
        
    data = torch.stack(data_list).permute(1, 0, 2)

    # NUTS VERSION
    from pyro.infer import MCMC, NUTS
    network = custom_torch_net
    num_chains = 4

    nuts_kernel = NUTS(hierarchical_ddm_model,
                       step_size = 0.01,
                       max_tree_depth = 5)
                       #jit_compile = True,
                       #ignore_jit_warnings = True)
                       #max_tree_depth = 1)
    
    mcmc = MCMC(nuts_kernel, 
                num_samples = 100, 
                warmup_steps = 100, 
                num_chains = num_chains, 
                initial_params = {'v_mu_mu': torch.tensor(param_dict['v_mu']).repeat(num_chains, 1),
                                  'v_mu_std': torch.tensor(param_dict['v_std']).repeat(num_chains, 1),
                                  'a_mu_mu': torch.tensor(param_dict['a_mu']).repeat(num_chains, 1),
                                  'a_mu_std': torch.tensor(param_dict['a_std']).repeat(num_chains, 1),
                                  'z_mu_mu': torch.tensor(param_dict['z_mu']).repeat(num_chains, 1),
                                  'z_mu_std': torch.tensor(param_dict['z_std']).repeat(num_chains, 1),
                                  't_mu_mu': torch.tensor(param_dict['t_mu']).repeat(num_chains, 1),
                                  't_mu_std': torch.tensor(param_dict['t_std']).repeat(num_chains, 1),
                                  'v_subj': torch.tensor(param_dict['v_subj']).repeat(num_chains, 1),
                                  'a_subj': torch.tensor(param_dict['a_subj']).repeat(num_chains, 1),
                                  'z_subj': torch.tensor(param_dict['z_subj']).repeat(num_chains, 1),
                                  't_subj': torch.tensor(param_dict['t_subj']).repeat(num_chains, 1),
                                  }
               )
    

    start_t = time()
    mcmc.run(n_subjects, n_samples_by_subject, data)
    end_t = time()
    
    # Make arviz data
    az_mcmc = az.from_pyro(mcmc)
    az_mcmc.posterior.attrs['runtime'] = end_t - start_t
    
    # Save arviz data:
    my_uuid = uuid.uuid1().hex
    pickle.dump(az_mcmc, 
                open('/users/afengler/data/proj_lan_varinf/LAN_varinf/data/parameter_recovery/pyro_mcmc_' + my_uuid, 'wb'), 
                protocol = 3) 
    
    print('DONE')