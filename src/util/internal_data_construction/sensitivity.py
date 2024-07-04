
'''Module for analysing the sensitivity of a learned embedding in a Transformer to previous tokens in a sequence'''

import jax
import jax.numpy as jnp
from typing import List, Dict, Tuple, Optional
from ml_collections import config_dict

from src.data.datagenerator import DataGenerator
from src.training_init import TrainingInitializer

class SensitivityAnalyser:
    '''Class implementing the sensitivity analysis'''
    def __init__(self, 
                 model_config: config_dict.ConfigDict,
                 experiment_config: config_dict.ConfigDict, 
                 copy_list: Optional[List[any]] = None):
        '''
        Initializes the sensitivity analysis
        Args:
            'model_config' (config_dict.ConfigDict): Configuration for the model
            'experiment_config' (config_dict.ConfigDict): Configuration for the experiment
            'copy_list' Optional[List[any]]: Already computed sensitivities
        ''' 
        self.model_config = model_config
        self.experiment_config = experiment_config
        self.copy_list = copy_list

        (self.model, self.optimizer, self.data_generator, self.train_module) = TrainingInitializer(model_config, experiment_config).run()
    
    def compute_sensitivities(self, 
                              data_mean: List[jnp.ndarray], 
                              data_std: List[jnp.ndarray], 
                              token_id: int):
        '''
        Computes the sensitivity-mean,std and yavg lists for a given embedding-token.
        '''
        mean_results = []
        std_results = []
        yavg_results = []
        for i in reversed(range(max(0, token_id-self.experiment_config.experiment.analysis_range), token_id+1, 1)):
            ymean = [jax.device_get(point[i]) for point in data_mean]
            ystd = [jax.device_get(point[i]) for point in data_std]
            ystd_min = [float(ymean[k]) - float(ystd[k]) for k in range(len(ystd))]
            ystd_plus = [float(ymean[k]) + float(ystd[k]) for k in range(len(ystd))]
            mean_results.append(ymean)
            std_results.append(ystd) #(ystd_min, ystd_plus))
        yavg_list = []
        for i in range(0,max(token_id-self.experiment_config.experiment.analysis_range,0)):
            yavg_list.append([jax.device_get(point[i]) for point in data_mean])
        yavg = jnp.mean(jnp.array(yavg_list),axis=0)
        if token_id > 5:
            yavg_results.append(yavg)
        return mean_results, std_results, yavg_results
    
    def sensitivity_lists(self, 
                          copy_list: Optional[List[any]]) -> Tuple[List[any]]:
        '''
        Computes the mean, std sensitivity lists for the given sensitivity ('copy_list') over seeds,
        as well as the average values for sensitivity w.r.t. 
        further previous tokens outside of experiment range.
        '''
        copy_means = []
        copy_stds = []
        for i in range(len(copy_list[0])):
            ps = tuple([copy_list[seed][i] for seed in range(len(copy_list))])
            f_idx_mean = lambda idx: jnp.mean(jnp.array([p[idx] for p in ps]), axis=0)
            f_idx_std = lambda idx: jnp.std(jnp.array([p[idx] for p in ps]), axis=0)
            f1_mean,f25_mean,f50_mean = f_idx_mean(0), f_idx_mean(1), f_idx_mean(2)
            f1_std,f25_std,f50_std = f_idx_std(0), f_idx_std(1), f_idx_std(2)
            copy_means.append((f1_mean,f25_mean,f50_mean))
            copy_stds.append((f1_std,f25_std,f50_std))

        # Extract data
        cms, cmm, cml = tuple([[p[idx] for p in copy_means] for idx in range(3)])
        css, csm, csl = tuple([[p[idx] for p in copy_stds] for idx in range(3)])
        
        results_s = self.compute_sensitivities(cms, css, token_id=1)
        results_m = self.compute_sensitivities(cmm, csm, token_id=(self.experiment_config.data.seq_len-1)//2)
        results_l = self.compute_sensitivities(cml, csl, token_id=self.experiment_config.data.seq_len-1)
        
        return results_s, results_m, results_l
    
    def run(self) -> Dict[str, List[any]]:
        '''Runs the sensitivity analysis and returns the results as a dictionary.'''
        if self.copy_list:
            # No need to train models
            return {'sensitivity_lists': self.sensitivity_lists(copy_list=self.copy_list)}
        else:
            # Training models
            copy_list = []
            for seed in self.experiment_config.seeds:
                copy_list_seed = []
                rng = jax.random.PRNGKey(seed)
                rng, test_rng, train_rng = jax.random.split(rng, 3)
                state_tf, rng = self.train_module.get_init_state(rng, interpol_call=False)
                for epoch_idx in range(self.experiment_config.experiment.train_len):
                    state_tf, train_rng, _, _, copy = self.train_module.train_epoch(epoch=epoch_idx,
                                                                                    state=state_tf,
                                                                                    rng=train_rng,
                                                                                    test_rng=test_rng,
                                                                                    num_batches_train=100,
                                                                                    interpolate=False)
                    copy_list_seed.append(copy)
                copy_list.append(copy_list_seed)
            return {'sensitivity_lists': self.sensitivity_lists(copy_list=copy_list),
                    'copy_list':copy_list}


