
'''Module for probing of multi-layer Transformer models.'''

import jax.numpy as jnp
from jax import random, vmap
from flax import linen as nn
from flax.training import train_state
from ml_collections import config_dict
from typing import Callable, List, Tuple, Dict

from src.data.datagenerator import DataGenerator

class ProbeLayers():
    '''Class for probing of multi-layer Transformer models.'''
    def __init__(self,
                 experiment_config: config_dict.ConfigDict,
                 data_generator: DataGenerator,
                 model_tf: nn.Module,
                 state_tf: train_state.TrainState,
                 probes_list: List[str],
                 inv_fn: Callable[[jnp.ndarray], jnp.ndarray],
                 probe_lambda: float = 0.001,
                 partobs: bool = False,
                 partobs_range: int = 35,
                 use_mlp: bool = False,
                 clean_target_probe: bool = False,
                 target_probe_threshold: float = 2.0):
        '''Initializes the ProbeLayers class.
        Args:
            'experiment_config' (config_dict.ConfigDict): Configuration of the experiment.
            'data_generator' (DataGenerator): Data generator.
            'model_tf' (nn.Module): Transformer model.
            'state_tf' (train_state.TrainState): Train state.
            'probes_list' (List[str]): List of probes to run. Possible Probes: ('curr_probe', 'next_probe', 'inverse_curr_probe', 'implicit_target').
            'inv_fn' (Callable[[jnp.ndarray], jnp.ndarray]): Function to compute the inverse of a matrix.
        '''
        self.experiment_config = experiment_config
        self.data_generator = data_generator
        self.model_tf = model_tf
        self.state_tf = state_tf  
        self.probes_list = probes_list
        self.inv_fn = inv_fn
        self.probe_lambda = probe_lambda
        self.partobs = partobs
        self.partobs_range = partobs_range
        self.use_mlp = use_mlp
        self.clean_target_probe = clean_target_probe
        self.target_probe_threshold = target_probe_threshold
    
    def get_features(self, batch: jnp.ndarray) -> jnp.ndarray:
        feature_seq_func = lambda seq : vmap(self.data_generator._mini_mlp)(seq)
        feature_batch = vmap(feature_seq_func, in_axes=(0))(batch)
        return feature_batch / (jnp.linalg.norm(feature_batch, axis=-1)[...,None] + 1e-16)

    def clean_batch(self, 
                    obs: Tuple[jnp.ndarray, jnp.ndarray],
                    probe_token: int, 
                    probe_lambda: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        '''Cleans the batch by removing norm outliers.'''
        obs_d, obs_l = obs
        eye_dim = self.partobs_range if self.partobs else self.data_generator.get_data_info()['obs_dim']
        if self.data_generator.get_data_info()['constr']:
            dim_constr = self.data_generator.get_data_info()['data_dim']
            XXtinv = self.inv_fn((jnp.transpose(obs_d[:,:probe_token,dim_constr:2*dim_constr], axes=(0,2,1)))@(obs_d[:,:probe_token,dim_constr:2*dim_constr]) + probe_lambda*jnp.eye(dim_constr))
            dataXXtinv = vmap(jnp.matmul, in_axes=(0,0))(obs_d[:,probe_token,dim_constr:2*dim_constr], XXtinv)
        else:
            XXtinv = self.inv_fn((jnp.transpose(obs_d[:,:probe_token,:], axes=(0,2,1)))@(obs_d[:,:probe_token,:]) + probe_lambda*jnp.eye(eye_dim))
            dataXXtinv = vmap(jnp.matmul, in_axes=(0,0))(obs_d[:,probe_token,:], XXtinv)
        mean_norm = jnp.mean(jnp.linalg.norm(dataXXtinv, axis=1),axis=0)
        std_norm = jnp.std(jnp.linalg.norm(dataXXtinv, axis=1),axis=0)
        def is_norm_less_than_t(vector):
            return (jnp.linalg.norm(vector)-mean_norm)/std_norm < self.experiment_config.experiment.batch_norm_threshold
        mask = vmap(is_norm_less_than_t, in_axes=(0))(dataXXtinv)
        return obs_d[mask], obs_l[mask], mask
    
    def clean_batch_target(self,
                           obs: Tuple[jnp.ndarray, jnp.ndarray], 
                           probe_token: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        _, obs_l = obs
        def contains_probe_token(vector):
            return jnp.any(vector >= self.target_probe_threshold)
        mask = vmap(contains_probe_token)(obs_l[:, probe_token])
        filtered_obs_l = obs_l[~mask]
        return filtered_obs_l, ~mask
        

    def probe(self, 
              seed:int, 
              probe_fn:Callable[[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray], int], List[List[float]]]) -> List[List[float]]:
        '''Runs the probe.'''
        test_rng = random.PRNGKey(seed=seed)
        obs, _ = self.data_generator.get_data(rng=test_rng,
                                              batch_size=self.experiment_config.data.test_batch_size)
        _, (activations, _, _, _) = self.model_tf.apply({'params': self.state_tf.params}, obs[0], interpol_call=False)
        
        if self.partobs:
            obs_d, obs_l = obs
            batch_size, seq_len, obs_dim = obs_d.shape
            embed_dim = self.partobs_range
            constructed_data = jnp.zeros(shape=(batch_size, seq_len, embed_dim))
            constructed_data = constructed_data.at[:,:,0:obs_dim].set(obs_d)
            for k in range(1, embed_dim // obs_dim):
                shifted_data = jnp.concatenate((jnp.zeros(shape=(batch_size,(k),obs_dim)),obs_d[:,:-1*(k),:]),axis=1)
                constructed_data = constructed_data.at[:,:,k*obs_dim:(k+1)*obs_dim].set(shifted_data)
            obs_d = constructed_data
            obs = (obs_d, obs_l)
        elif self.use_mlp:
            obs_d, obs_l = obs
            nonlin_features = self.get_features(obs_d)
            obs = (nonlin_features, obs_l)
        bs,_,_ = obs[0].shape
        
        return probe_fn(activations, obs, bs)
    
    def curr_probe_fn(self, 
                      activations: List[jnp.ndarray], 
                      obs: Tuple[jnp.ndarray, jnp.ndarray],
                      bs: int) -> List[List[float]]:
        '''Probes the embedding at timestep i for the data token at timestep i.'''
        (obs_d, _) = obs
        self_results = []
        for probe_target in range(self.experiment_config.data.seq_len):
            token_probe = []
            for i in range(len(activations)):
                last_activation_tok = activations[i][:,probe_target,:]
                if self.data_generator.get_data_info()['constr']:
                    dim_constr = self.data_generator.get_data_info()['data_dim']
                    target = obs_d[:,probe_target,dim_constr:2*dim_constr]
                else:
                    target = obs_d[:,probe_target,:]
                k = jnp.linalg.lstsq(last_activation_tok,target)
                token_probe.append(jnp.sum((last_activation_tok@k[0] - target)**2)/(2*bs))
            self_results.append(token_probe)
        return self_results
    
    def next_probe_fn(self, 
                      activations: List[jnp.ndarray], 
                      obs: Tuple[jnp.ndarray, jnp.ndarray],
                      bs: int) -> List[List[float]]:
        '''Probes the embedding at timestep i for the data token at timestep i+1.'''
        (_, obs_l) = obs
        next_results = []
        for probe_target in range(self.experiment_config.data.seq_len):
            if self.clean_target_probe:
                obs_l, mask = self.clean_batch_target(obs, probe_target)
                bs = jnp.sum(mask)
            token_probe = []
            for i in range(len(activations)):
                if self.clean_target_probe:
                    last_activation_tok = activations[i][:,probe_target,:][mask]
                else:
                    last_activation_tok = activations[i][:,probe_target,:]
                target = obs_l[:,probe_target,:]
                k = jnp.linalg.lstsq(last_activation_tok,target)
                token_probe.append(jnp.sum((last_activation_tok@k[0] - target)**2)/(2*bs))
            next_results.append(token_probe)
        return next_results

    def inverse_curr_probe(self, 
                           activations: List[jnp.ndarray], 
                           obs: Tuple[jnp.ndarray, jnp.ndarray],
                           bs: int) -> List[List[float]]:
        '''Probes the embedding at timestep i for the data token * inverse at timestep i.'''
        inverse_curr_results = []
        inv_probe_lam = self.probe_lambda
        for probe_target in range(self.experiment_config.data.seq_len):
            # clean data:
            obs_d, _, mask = self.clean_batch(obs, probe_target, inv_probe_lam)
            bs = jnp.sum(mask)
            # compute inv:
            if self.data_generator.get_data_info()['constr']:
                dim_constr = self.data_generator.get_data_info()['data_dim']
                XXtinv = self.inv_fn((jnp.transpose(obs_d[:,:probe_target,dim_constr:2*dim_constr], axes=(0,2,1)))@(obs_d[:,:probe_target,dim_constr:2*dim_constr]) + inv_probe_lam*jnp.eye(dim_constr))
                dataXXtinv = vmap(jnp.matmul, in_axes=(0,0))(obs_d[:,probe_target,dim_constr:2*dim_constr], XXtinv)
            else:
                eye_dim = self.partobs_range if self.partobs else self.data_generator.get_data_info()['obs_dim']
                XXtinv = self.inv_fn((jnp.transpose(obs_d[:,:probe_target,:], axes=(0,2,1)))@(obs_d[:,:probe_target,:]) + inv_probe_lam*jnp.eye(eye_dim))
                dataXXtinv = vmap(jnp.matmul, in_axes=(0,0))(obs_d[:,probe_target,:], XXtinv)
            token_probe = []
            for i in range(len(activations)):
                last_activation_tok = activations[i][:,probe_target,:][mask]
                target = dataXXtinv
                k = jnp.linalg.lstsq(last_activation_tok,target)
                token_probe.append(jnp.sum((last_activation_tok@k[0] - target)**2)/(2*bs))
            inverse_curr_results.append(token_probe)
        return inverse_curr_results
    
    def control_probe_fn(self, 
                         activations: List[jnp.ndarray], 
                         obs: Tuple[jnp.ndarray, jnp.ndarray],
                         bs: int) -> List[List[float]]:
        '''Control Probe for inverse-probing.'''
        inverse_curr_results = []
        inv_probe_lam = self.probe_lambda
        for probe_target in range(self.experiment_config.data.seq_len):
            # clean data:
            obs_d, obs_l, mask = self.clean_batch(obs, probe_target, inv_probe_lam)
            bs = jnp.sum(mask)
            # compute inv:
            if self.data_generator.get_data_info()['constr']:
                dim_constr = self.data_generator.get_data_info()['data_dim']
                XXtinv = self.inv_fn((jnp.transpose(obs_d[:,:probe_target,dim_constr:2*dim_constr], axes=(0,2,1)))@(obs_d[:,:probe_target,dim_constr:2*dim_constr]) + inv_probe_lam*jnp.eye(dim_constr))
                dataXXtinv = vmap(jnp.matmul, in_axes=(0,0))(obs_d[:,probe_target,dim_constr:2*dim_constr], XXtinv)
            else:
                eye_dim = self.partobs_range if self.partobs else self.data_generator.get_data_info()['obs_dim']
                XXtinv = self.inv_fn((jnp.transpose(obs_d[:,:probe_target,:], axes=(0,2,1)))@(obs_d[:,:probe_target,:]) + inv_probe_lam*jnp.eye(eye_dim))
                dataXXtinv = vmap(jnp.matmul, in_axes=(0,0))(obs_d[:,probe_target,:], XXtinv)
            token_probe = []
            for i in range(len(activations)):
                target = obs_l[:,probe_target,:]
                k = jnp.linalg.lstsq(dataXXtinv, target)
                token_probe.append(jnp.sum((dataXXtinv@k[0] - target)**2)/(2*bs))
            inverse_curr_results.append(token_probe)
        return inverse_curr_results

    def implicit_target_probe(self, 
                              activations: List[jnp.ndarray], 
                              obs: Tuple[jnp.ndarray, jnp.ndarray],
                              bs: int) -> List[List[float]]:
        '''Probes the embedding at timestep i for the data token * implicit target at timestep i.'''
        implicit_target_results = []
        gd_learning_rate = 0.45
        model_learning_rate = 0.5
        inv_probe_lam = self.experiment_config.experiment.inv_probe_lambda
        for probe_target in range(self.experiment_config.data.seq_len):
            # clean data:
            obs_d, obs_l, mask = self.clean_batch(obs, probe_target, inv_probe_lam)
            bs = jnp.sum(mask)
            # compute inv:
            eye_dim = self.partobs_range if self.partobs else self.data_generator.get_data_info()['obs_dim']
            XXtinv = self.inv_fn((jnp.transpose(obs_d[:,:probe_target,:], axes=(0,2,1)))@(obs_d[:,:probe_target,:]) + inv_probe_lam*jnp.eye(eye_dim))
            dataXXtinv = vmap(jnp.matmul, in_axes=(0,0))(obs_d[:,probe_target,:], XXtinv)
            XtY = (jnp.transpose(obs_d[:,:probe_target,:], axes=(0,2,1))) @ obs_d[:,1:probe_target+1,:]
            
            token_probe = []
            k1_gd = jnp.linalg.lstsq(obs_d[:,probe_target,:], dataXXtinv)
            preds_gd = gd_learning_rate*(vmap(jnp.matmul, in_axes=(0,0))((obs_d[:,probe_target,:]@k1_gd[0]),XtY))
            loss_gd = jnp.sum((preds_gd - obs_l[:,probe_target,:])**2)/(2*bs)
            token_probe.append(loss_gd)

            for i in range(len(activations)):
                last_activation_tok = activations[i][:,probe_target,:][mask]
                k1_tf = jnp.linalg.lstsq(last_activation_tok, dataXXtinv)
                preds_tf = model_learning_rate*(vmap(jnp.matmul, in_axes=(0,0))((last_activation_tok@k1_tf[0]),XtY))
                loss_tf = jnp.mean(jnp.square(preds_tf - obs_l[:,probe_target,:]))
                token_probe.append(loss_tf)
            implicit_target_results.append(token_probe)
        return implicit_target_results


    def run(self) -> Dict[str, any]:
        '''Runs the probe for all the probes in the list.'''
        probe_len = range(len(self.probes_list))
        results = [[] for _ in range(5)]

        # TODO: Currently, if e.g. 'curr_probe' not incldued, index error

        for probe in self.probes_list:
            for seed in self.experiment_config.seeds:
                if probe == 'curr_probe':
                    results[0].append(self.probe(seed, self.curr_probe_fn))
                elif probe == 'next_probe':
                    results[1].append(self.probe(seed, self.next_probe_fn))
                elif probe == 'inverse_curr_probe':
                    results[2].append(self.probe(seed, self.inverse_curr_probe))
                elif probe == 'implicit_target':
                    results[3].append(self.probe(seed, self.implicit_target_probe))
                elif probe == 'control_probe':
                    results[4].append(self.probe(seed, self.control_probe_fn))
                else:
                    raise ValueError('Invalid probe')
        return {'results': results}
        
            