
'''Module for the Interpolator class. This class is used to run the interpolation experiments.'''

import jax
import jax.numpy as jnp
from jax import random
from tqdm import tqdm
from typing import List, Dict
from ml_collections import config_dict

from src.training_init import TrainingInitializer
from src.util.interpolation.interpol_helpers import OneLayerRevAlg_Linear, WeightGenerator, TFSubdiagonals

class Interpolator:
    '''Class implementing the Interpolator.'''
    def __init__(
        self,
        model_config: config_dict.ConfigDict,
        experiment_config: config_dict.ConfigDict,
    ):
        '''
        Initializes the Interpolator.
        Args:
            model_config: The model configuration.
            experiment_config: The experiment configuration.
        '''
        self.model_config = model_config
        self.experiment_config = experiment_config

        self.is_linear = self.model_config.linear
        self.peak_lr = self.experiment_config.optim.peak_lr
        self.init_train_len = self.experiment_config.experiment.init_train_len
        self.interpol_retrain_len = self.experiment_config.experiment.interpol_retrain_len

        self.seeds = self.experiment_config.seeds
        self.num_layers = self.model_config.num_layers

        self.tf_losses_multiseed = []
        self.interpolation_losses_multiseed = []
        self.result_revAlg_losses = []
        self.tf_params = []
        
        self.weight_generator = WeightGenerator(data_dim=self.experiment_config.data.data_dim)
        _,_, self.data_generator, self.train_module = TrainingInitializer(model_config, experiment_config).run()
        
        if self.num_layers == 1:
            self.rev_alg_module = OneLayerRevAlg_Linear(data_dim=self.experiment_config.data.data_dim, 
                                                        seq_len=self.experiment_config.data.data_dim)

    def run(self) -> Dict[str, any]:
        '''
        Runs the interpolation experiment.
        Returns:
            A dictionary containing the results of the experiment.
        '''
        for seed in self.seeds:
            rng = jax.random.PRNGKey(seed)
            rng, test_rng, train_rng = jax.random.split(rng, 3)
            state_tf, rng = self.train_module.get_init_state(rng, interpol_call=True)
            for epoch_idx in range(self.init_train_len):
                state_tf, train_rng, _, _, _ = self.train_module.train_epoch(epoch=epoch_idx,
                                                                                state=state_tf,
                                                                                rng=train_rng,
                                                                                test_rng=test_rng,
                                                                                num_batches_train=100,
                                                                                interpolate=False)
            
            if self.num_layers == 1:
                if self.is_linear:
                    diags = TFSubdiagonals.two_head_diags(params=state_tf.params, key_size=20)
                    flattened_args = tuple(element for sub in diags for element in sub)
                    revAlg_loss = self.rev_alg_module.evaluate_revAlg(flattened_args,test_rng,self.data_generator)
                    qfm,kfm,vfm,pfm = self.weight_generator.two_head_ks20(*flattened_args)
                    interpol_kernels = [qfm],[kfm],[vfm],[pfm]
                else:
                    diags = TFSubdiagonals.two_head_diags(params=state_tf.params, key_size=20)
                    flattened_args = tuple(element for sub in diags for element in sub)
                    qfm,kfm,vfm,pfm = self.weight_generator.two_head_ks20(*flattened_args)
                    interpol_kernels = [qfm],[kfm],[vfm],[pfm]
                    revAlg_params = self.train_module.replace_all_weights(state_tf.params,interpol_kernels)
                    revAlg_loss = self.train_module.fast_pure_test_computation(revAlg_params, test_rng)
            else:
                diags = TFSubdiagonals.four_head_diags_multilayer(params=state_tf.params, num_layers=self.num_layers, key_size=40)
                qfmI,kfmI,vfmI,pfmI = [],[],[],[]
                for lay_num in range(self.num_layers):
                    ((d1,d2,d3,d4),(p1,p2,p3,p4)) = diags[lay_num]
                    qfm_layer,kfm_layer,vfm_layer,pfm_layer = self.weight_generator.four_head_ks40(d1,d2,d3,d4,p1,p2,p3,p4,data_dim=10)
                    qfmI.append(qfm_layer)
                    kfmI.append(kfm_layer)
                    vfmI.append(vfm_layer)
                    pfmI.append(pfm_layer)
                interpol_kernels = qfmI,kfmI,vfmI,pfmI
                revAlg_params = self.train_module.replace_all_weights(state_tf.params,interpol_kernels)
                revAlg_loss = self.train_module.fast_pure_test_computation(revAlg_params, test_rng)
            
            self.result_revAlg_losses.append(revAlg_loss)

            rng = jax.random.PRNGKey(seed)
            rng, test_rng, train_rng = jax.random.split(rng, 3)
            state_interpol, rng = self.train_module.get_init_state(rng, interpol_call=True)
 
            tf_losses = []
            ip_losses = []

            for epoch_idx in range(self.interpol_retrain_len):
                state_interpol, train_rng, test_loss, ip_loss, _ = self.train_module.train_epoch(epoch=epoch_idx,
                                                                                        state=state_interpol,
                                                                                        rng=train_rng,
                                                                                        test_rng=test_rng,
                                                                                        num_batches_train=100,
                                                                                        interpolate=True,
                                                                                        interpol_kernels=interpol_kernels)
                tf_losses.append(test_loss)
                ip_losses.append(ip_loss)

            self.tf_losses_multiseed.append(tf_losses)
            self.interpolation_losses_multiseed.append(ip_losses)
            self.tf_params.append(state_tf.params)
        
        return {'tf_losses_multiseed':self.tf_losses_multiseed,
                'interpolation_losses_multiseed':self.interpolation_losses_multiseed,
                'revAlg_losses':self.result_revAlg_losses,
                'tf_params':self.tf_params}