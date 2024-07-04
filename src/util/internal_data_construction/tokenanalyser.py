
'''Module implementing the token analysis for the partially observed data setting.'''

import jax.numpy as jnp
import flax.linen as nn
from ml_collections import config_dict
from typing import List, Dict
from flax.training import train_state
from jax import random, vmap

from src.data.datagenerator import DataGenerator

class TokenAnalyser:
    '''
    Class implementing the token analysis for the partially observed data setting.
    That is, analysing the sensitivity of a learned embedding in a Transformer in the partially observed setting
    to previous tokens in a sequence, compared across different models.
    '''
    def __init__(self,
                 models: List[nn.Module],
                 states: List[train_state.TrainState],
                 data_generator: DataGenerator,
                 experiment_config: config_dict.ConfigDict,
                 nonlinear: bool = False,
                 r2: bool = False,
                 layer: int = 1):
        '''
        Initializes the token analysis
        Args:
            'models' (List[nn.Module]): List of models to be analysed
            'states' (List[train_state.TrainState]): List of states of the models
            'data_generator' (DataGenerator): Data generator for the analysis
            'experiment_config' (config_dict.ConfigDict): Configuration for the experiment
            'nonlinear' (bool): Nonlinear feature probing
            'r2' (bool): Whether to use R2 score
            'layer' (int): The layer to probe
        '''
        
        self.models = models
        self.states = states
        self.data_generator = data_generator
        self.experiment_config = experiment_config
        self.nonlinear = nonlinear
        self.r2 = r2
        self.layer = layer

    def r2_score_batch(self, y_true, y_pred):
        y_true = jnp.array(y_true)
        y_pred = jnp.array(y_pred)
        y_mean = jnp.mean(y_true, axis=0)
        ss_tot = jnp.sum((y_true - y_mean) ** 2, axis=0)
        ss_res = jnp.sum((y_true - y_pred) ** 2, axis=0)
        r2 = 1 - (ss_res / ss_tot)
        return jnp.mean(r2)
    
    def run(self, token:int) -> Dict[str, List[jnp.ndarray]]:
        '''Runs the token-analysis for the given token'''
        results = []
        for seed in self.experiment_config.seeds:
            seed_results = []
            test_rng = random.PRNGKey(seed)
            obs_data, _ = self.data_generator.get_data(rng=test_rng, 
                                                       batch_size=self.experiment_config.data.test_batch_size)
            data, _ = obs_data
            for model, state in zip(self.models, self.states):
                _, (activations, _, mlp_outs, _) = model.apply({'params': state.params}, data, interpol_call=False)
                result_model = []
                for probe_token in range(token-self.experiment_config.experiment.probe_range, token+1):
                    last_activation_tok = activations[self.layer][:,token,:]
                    if self.nonlinear:
                        prob = vmap(self.data_generator._mini_mlp, in_axes=(0))(data[:,probe_token,:])
                    else:
                        prob = data[:,probe_token,:]
                    k = jnp.linalg.lstsq(last_activation_tok,prob)
                    loss = self.r2_score_batch(prob, last_activation_tok@k[0]) if self.r2 else jnp.mean((last_activation_tok@k[0] - prob)**2)
                    result_model.append(loss)
                seed_results.append(result_model)
            results.append(seed_results)
        
        seed_res_arr = jnp.array(results)
        results_means = list(jnp.mean(seed_res_arr,axis=0))
        results_svs = list(jnp.std(seed_res_arr,axis=0))

        return {'results_means': results_means,
                'results_svs': results_svs}