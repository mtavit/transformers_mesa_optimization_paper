
'''Module for probing nonlinear features in a transfomer model.'''

import jax
import jax.numpy as jnp
from typing import Dict, List

from src.training_init import TrainingInitializer

class NonlinearFeatureProbe:
    '''Class for probing nonlinear features in a transformer model.'''
    def __init__(self,
                experiment_config,
                model_config):
        '''
        Initializes the nonlinear feature probe
        Args:
            experiment_config: Configuration for the experiment
            model_config: Configuration for the model
        '''
        self.experiment_config = experiment_config
        self.model_config = model_config
        self.seeds = self.experiment_config.seeds

        (self.model_tf, self.optimizer, self.data_generator, self.train_module) = TrainingInitializer(model_config, experiment_config).run()
        
    
    def batch_mlp(self, batch: jnp.ndarray) -> jnp.ndarray:
        '''Returns MLP features for the batch'''
        part_mini_mlp = self.data_generator._mini_mlp
        feature_seq_func = lambda seq : jax.vmap(part_mini_mlp)(seq)
        feature_batch = jax.vmap(feature_seq_func, in_axes=(0))(batch)
        return feature_batch / (jnp.linalg.norm(feature_batch, axis=-1)[...,None] + 1e-16)
    
    def r2_score_batch(self, y_pred, y_true):
        y_true = jnp.array(y_true)
        y_pred = jnp.array(y_pred)
        y_mean = jnp.mean(y_true, axis=0)
        ss_tot = jnp.sum((y_true - y_mean) ** 2, axis=0)
        ss_res = jnp.sum((y_true - y_pred) ** 2, axis=0)
        r2 = 1 - (ss_res / ss_tot)
        return jnp.mean(r2)

    def run(self, token: int) -> Dict[str, List[List[jnp.ndarray]]]:
        '''Runs the nonlinear feature probe'''
        results = []
        models = []
        all_out_probes = []
        for seed in self.experiment_config.seeds:
            # Train Model:
            rng = jax.random.PRNGKey(seed)
            rng, test_rng, train_rng = jax.random.split(rng, 3)
            state_tf, rng = self.train_module.get_init_state(rng, interpol_call=False)
            (data, _), _ = self.data_generator.get_data(rng=test_rng, batch_size=self.experiment_config.data.test_batch_size)
            out_probes = []
            for epoch_idx in range(self.experiment_config.experiment.train_len):
                if epoch_idx % 10 == 0:
                    _, (activations, _, mlp_outputs, _) = self.model_tf.apply({'params': state_tf.params}, data, interpol_call=False)
                    out_probes.append(activations)
                state_tf, train_rng, _, _, _ = self.train_module.train_epoch(epoch=epoch_idx,
                                                                                state=state_tf,
                                                                                rng=train_rng,
                                                                                test_rng=test_rng,
                                                                                num_batches_train=100,
                                                                                interpolate=False)
            all_out_probes.append(out_probes)
            # Probe Features per Layer:
            seed_results = [[] for _ in range(len(out_probes[0]))]
            for layer_idx in range(len(out_probes[0])):    
                for train_step in range(len(out_probes)):
                    step_probes = out_probes[train_step]

                    prob = jax.vmap(self.data_generator._mini_mlp, in_axes=(0))(data[:,token,:])
                    k = jnp.linalg.lstsq(step_probes[layer_idx][:,token,:], prob)
                    score = self.r2_score_batch((step_probes[layer_idx][:,token,:])@k[0], prob)
                    seed_results[layer_idx].append(score)
                    
            results.append(seed_results)
            models.append(state_tf.params)
        
        return {'results': results, 'models': models}