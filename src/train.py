
'''Module for transformer model training, including methods for interpolating models.'''

import flax
import optax
import jax.numpy as jnp
from jax import random, vmap, grad, value_and_grad, jit, lib
from flax.training import train_state
from flax import traverse_util
from functools import partial
from typing import Tuple, List

from src.data.datagenerator import DataGenerator

def _compute_loss(preds: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
        '''Computes the mean squared error (MSE) loss for a batch.'''
        assert preds.shape == targets.shape
        bs, sl, _ = preds.shape
        return (jnp.sum((targets -preds)**2)/(2*bs*sl))

class Training:
    '''Class for training the transformer model.'''
    def __init__(
        self,
        model: flax.linen.Module,
        optimizer: optax.GradientTransformation,
        data_generator: DataGenerator,
        interpolate: bool,
        sensitivity_analysis: bool,
        batch_size: int,
        test_batch_size: int
    ):
        '''
        Initializes the training class with the specified parameters.
        Args:
            'model' (flax.linen.Module): The transformer model.
            'optimizer' (optax.GradientTransformation): The optimizer.
            'data_generator' (DataGenerator): The data generator.
            'interpolate' (bool): Flag whether to perform interpolation during evaluation.
            'sensitivity_analysis' (bool): Flag whether to perform sensitivity analysis during training.
            'batch_size' (int): The batch size for training.
            'test_batch_size' (int): The batch size for testing.
        '''
        self.model = model
        self.optimizer = optimizer
        self.data_generator = data_generator
        self.interpolate = interpolate
        self.sensitivity_analysis = sensitivity_analysis
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.obs_dim = data_generator.get_data_info()['obs_dim']

    def get_init_state(self, 
                       rng: random.PRNGKey, 
                       interpol_call: bool) -> Tuple[train_state.TrainState, random.PRNGKey]:
        '''
        Initializes the training state with the specified random number generator key and interpolation flag.
        Args:
            'rng' (jax.random.PRNGKey): The random number generator key.
            'interpol_call' (bool): Flag whether to perform interpolation during evaluation.
        '''
        rng, ex_rng, init_rng = random.split(rng, 3)
        (exmp_inp, _), _ = self.data_generator.get_data(rng=ex_rng, batch_size=self.batch_size)
        params = self.model.init({'params': init_rng}, exmp_inp, interpol_call=interpol_call)['params']
        state_init = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=self.optimizer)
        return state_init, rng

    def batch_to_input(self, batch: Tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        '''Extracts the input data from the batch.'''
        data, _ = batch
        return data

    def calculate_loss(self, 
                       params: flax.core.frozen_dict.FrozenDict, 
                       batch: Tuple[jnp.ndarray, jnp.ndarray],
                       interpol_call: bool) -> Tuple[jnp.ndarray, Tuple[any]]:
        '''
        Calculates the differentiable loss function.
        Args:
            'params' (flax.core.frozen_dict.FrozenDict): The model parameters.
            'batch' (Tuple[jnp.ndarray, jnp.ndarray]): The input batch.
            'interpol_call' (bool): Flag indicating whether to perform interpolation during evaluation.
        Returns:
            Tuple[jnp.ndarray, Tuple[any]] 'loss' (jnp.ndarray): The computed loss and 'tf_data' (Tuple[any]): The data from the transformer forward pass.
        '''
        inp_data, labels = batch
        logits, tf_data = self.model.apply({'params': params}, inp_data, interpol_call=interpol_call)
        preds = logits[:, :, :self.obs_dim]
        loss = _compute_loss(preds=preds, targets=labels)
        return loss, tf_data

    @partial(jit, static_argnums=(0, 3))
    def fast_train_step(self, 
                        state: train_state.TrainState, 
                        batch: Tuple[jnp.ndarray, jnp.ndarray],
                        interpol_call: bool) -> Tuple[train_state.TrainState, jnp.ndarray, Tuple[any]]:
        '''
        Performs a single training step.
        Args:
            'state' (flax.training.train_state.TrainState): The current training state.
            'batch' (Tuple[jnp.ndarray, jnp.ndarray]): The input batch.
            'interpol_call' (bool): Flag indicating whether to perform interpolation during evaluation.
        Returns:
            Tuple[train_state.TrainState, jnp.ndarray, Tuple[any]]:
                - 'state' (flax.training.train_state.TrainState): The updated training state, 
                - 'loss' (jnp.ndarray): The computed loss, 
                - 'tf_data' (Tuple[any]): The data from the transformer forward pass.
        '''
        loss_fn = lambda params: self.calculate_loss(params=params, batch=batch, interpol_call=interpol_call)
        (loss, (tf_data)), grads = value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss, tf_data

    @partial(jit, static_argnums=(0))
    def fast_eval_step(self, 
                       params: flax.core.frozen_dict.FrozenDict, 
                       batch: Tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        '''
        Performs a single evaluation step.
        Args:
            'params' (flax.core.frozen_dict.FrozenDict): The model parameters.
            'batch' (Tuple[jnp.ndarray, jnp.ndarray]): The input batch.
        Returns:
            'loss' (jnp.ndarray): The computed test loss.
        '''
        loss, _ = self.calculate_loss(params=params, batch=batch, interpol_call=False)
        return loss

    def weight_interpol(self, 
                        params: flax.core.frozen_dict.FrozenDict, 
                        interpol_kernels: Tuple[List[jnp.ndarray]]) -> flax.core.frozen_dict.FrozenDict:
        '''
        Interpolates the weights of the model.
        Args:
            'params' (flax.core.frozen_dict.FrozenDict): The model parameters.
            'qfm' (List[jnp.ndarray]): The query matrices.
            'kfm' (List[jnp.ndarray]): The key matrices.
            'vfm' (List[jnp.ndarray]): The value matrices.
            'pfm' (List[jnp.ndarray]): The projection matrices.
        Returns:
            'params_frozen' (flax.core.frozen_dict.FrozenDict): Transformer parameters where resp. interpol. weights contain user defined args.
        '''
        flat_params = traverse_util.flatten_dict(params)
        qfm, kfm, vfm, pfm = interpol_kernels
        for i in range(self.model.num_layers):
            flat_params[('tf_block', f'blocklist_{i}', 'self_attn', 'q_fm', 'kernel')] = qfm[i]
            flat_params[('tf_block', f'blocklist_{i}', 'self_attn', 'k_fm', 'kernel')] = kfm[i]
            flat_params[('tf_block', f'blocklist_{i}', 'self_attn', 'v_fm', 'kernel')] = vfm[i]
            flat_params[('tf_block', f'blocklist_{i}', 'self_attn', 'o_fm', 'kernel')] = pfm[i]
        params_frozen = traverse_util.unflatten_dict(flat_params)
        return params_frozen
    
    def replace_one_layer(self, 
                          params: flax.core.frozen_dict.FrozenDict, 
                          interpol_kernels: Tuple[jnp.ndarray],
                          i: int,
                          hybrid: bool = False) -> flax.core.frozen_dict.FrozenDict:
        '''
        Replaces the weights of one layer in the model. For multi-layer linearization.
        Args:
            'params' (flax.core.frozen_dict.FrozenDict): The model parameters.
            'qfm' (jnp.ndarray): The query matrix.
            'kfm' (jnp.ndarray): The key matrix.
            'vfm' (jnp.ndarray): The value matrix.
            'pfm' (jnp.ndarray): The projection matrix.
            'i' (int): Layer index.
            'hybrid' (bool): Flag, model has hybrid layer at position i.
        Returns:
            'params_frozen' (flax.core.frozen_dict.FrozenDict): Transformer parameters where layer_i weights contain user defined args.
        '''
        qfm, kfm, vfm, pfm = interpol_kernels
        flat_params = traverse_util.flatten_dict(params)
        if hybrid and i == 0:
            flat_params[('hybrid_block', 'self_attn', 'q_proj', 'kernel')] = qfm
            flat_params[('hybrid_block', 'self_attn', 'k_proj', 'kernel')] = kfm
            flat_params[('hybrid_block', 'self_attn', 'v_proj', 'kernel')] = vfm
            flat_params[('hybrid_block', 'self_attn', 'o_proj', 'kernel')] = pfm
        else:
            if hybrid:
                i -= 1
            flat_params[('tf_block', f'blocklist_{i}', 'self_attn', 'q_proj', 'kernel')] = qfm
            flat_params[('tf_block', f'blocklist_{i}', 'self_attn', 'k_proj', 'kernel')] = kfm
            flat_params[('tf_block', f'blocklist_{i}', 'self_attn', 'v_proj', 'kernel')] = vfm
            flat_params[('tf_block', f'blocklist_{i}', 'self_attn', 'o_proj', 'kernel')] = pfm
        return traverse_util.unflatten_dict(flat_params)

    def replace_all_weights(self, 
                            params: flax.core.frozen_dict.FrozenDict, 
                            interpol_kernels: Tuple[List[jnp.ndarray]]) -> flax.core.frozen_dict.FrozenDict:
        '''Replaces all weights in the model. For evaluation of RevAlg models.'''
        qfm, kfm, vfm, pfm = interpol_kernels
        flat_params = traverse_util.flatten_dict(params)
        for layer_index in range(self.model.num_layers):
            components_to_kernels = {
                'q': qfm[layer_index],
                'k': kfm[layer_index],
                'v': vfm[layer_index],
                'o': pfm[layer_index],
            }
            for component, kernel in components_to_kernels.items():
                base_key = ('tf_block', f'blocklist_{layer_index}', 'self_attn')
                flat_params[base_key + (f'{component}_fm', 'kernel')] = kernel
                flat_params[base_key + (f'{component}_proj', 'kernel')] = kernel
        params_frozen = traverse_util.unflatten_dict(flat_params)
        return params_frozen

    @partial(jit, static_argnums=(0))
    def fast_pure_test_computation(self, 
                                   params: flax.core.frozen_dict.FrozenDict, 
                                   test_rng: random.PRNGKey) -> jnp.ndarray:
        '''
        Performs a full evaluation computation for performance evaluation of RevAlgs.
        Args:
            'params' (flax.core.frozen_dict.FrozenDict): The model parameters.
            'test_rng' (jax.random.PRNGKey): The random number generator key for testing.
        Returns:
            'test_loss' (jnp.ndarray): The computed test loss.
        '''
        test_loss = 0
        for _ in range(10):
            test_rng, batch_rng = random.split(test_rng, 2)
            batch_TEST, _ = self.data_generator.get_data(rng=batch_rng, batch_size=self.test_batch_size)
            (step_loss, _) = self.calculate_loss(params=params, batch=batch_TEST, interpol_call=False)
            test_loss += step_loss
        return test_loss/10

    @partial(jit, static_argnums=(0))
    def fast_eval_step_interpol(self, 
                                params: flax.core.frozen_dict.FrozenDict, 
                                batch: Tuple[jnp.ndarray, jnp.ndarray],
                                interpol_kernels: Tuple[List[jnp.ndarray]]) -> jnp.ndarray:
        '''Performs a single evaluation step with weight interpolation.'''
        interpol_params = self.weight_interpol(params, interpol_kernels)
        (loss, _) = self.calculate_loss(params=interpol_params, batch=batch, interpol_call=True)
        return loss

    @partial(jit, static_argnums=(0))
    def fast_sensitivity(self, 
                         batch: Tuple[jnp.ndarray, jnp.ndarray],
                         state: train_state.TrainState) -> Tuple[List[jnp.ndarray]]:
        '''
        Performs sensitivity analysis.
        Args:
            'batch' (Tuple[jnp.ndarray, jnp.ndarray]): The input batch.
            'state' (flax.training.train_state.TrainState): The current training state.
        Returns:
            Tuple[List[jnp.ndarray]] ('listsnd', 'listmid', 'listlast'): The sensitivity analysis results for the second token, mid and last token in a sequence.
        '''       
        _,s,_ = batch[0].shape
        target_k = [1,(s-1)//2,s-1]
        res_list = []
        for k in target_k:
            grad_of_output_l_wrt_x = lambda l: vmap(grad(lambda x: self.model.apply({'params': state.params}, x[None, ...])[1][0][1][0][k][l],         #1: second output, 0: activations, 1:layer,  
                                                argnums=0))(batch[0])
            grads = vmap(lambda t: jnp.mean(jnp.linalg.norm(grad_of_output_l_wrt_x(t), axis=(2)),axis=0)[:k+1])(jnp.arange(self.model.embed_dim))
            grads_norm = jnp.mean(jnp.array(grads), axis=0)
            res_list.append(grads_norm)
        return tuple(res_list)

    def train_epoch(self, 
                    epoch: int, 
                    rng: random.PRNGKey, 
                    test_rng: random.PRNGKey, 
                    state: train_state.TrainState, 
                    num_batches_train: int, 
                    interpolate: bool,
                    interpol_kernels: Tuple[List[jnp.ndarray]] = None) -> Tuple[train_state.TrainState, random.PRNGKey, jnp.ndarray, jnp.ndarray, Tuple[List[jnp.ndarray]]]:
        '''
        Trains the model for a single epoch. While "epoch" is not the perfect term when used in our setting of online learning, it refers to a fixed number of train steps, here.
        Args:
            'epoch' (int): The current epoch.
            'rng' (jax.random.PRNGKey): The random number generator key.
            'test_rng' (jax.random.PRNGKey): The random number generator key for testing.
            'state' (flax.training.train_state.TrainState): The current training state.
            'num_batches_train' (int): The number of training batches.
            'interpolate' (bool): Flag whether to perform interpolation during evaluation.
            interp    (List[jnp.ndarray]): The query matrices for interpolation.
            'kfm' (List[jnp.ndarray]): The key matrices for interpolation.
            'vfm' (List[jnp.ndarray]): The value matrices for interpolation.
            'pfm' (List[jnp.ndarray]): The projection matrices for interpolation.
        Returns:
            Tuple[train_state.TrainState, random.PRNGKey, jnp.ndarray, jnp.ndarray, Tuple[List[jnp.ndarray]]]:
                - 'state' (flax.training.train_state.TrainState): The updated training state,
                - 'rng' (jax.random.PRNGKey): The updated random number generator key,
                - 'test_loss' (jnp.ndarray): The computed test loss,
                - 'interpolate_test_loss' (jnp.ndarray): The computed test loss with interpolation,
                - ('listsnd', 'listmid', 'listlast') (Tuple[List[jnp.ndarray]]): The sensitivity analysis results.
        '''
        rng, tr_rng = random.split(rng, 2)
        listsnd, listmid, listlast = [], [], []

        if self.sensitivity_analysis:
            batch_copy,_ = self.data_generator.get_data(rng=test_rng, batch_size=64)
            listsnd, listmid, listlast = self.fast_sensitivity(batch_copy, state)
        
        test_loss = 0
        interpolate_test_loss = 0
        for _ in range(10):
            test_rng, batch_rng = random.split(test_rng, 2)
            batch_TEST,_ = self.data_generator.get_data(rng=batch_rng, batch_size=self.test_batch_size)
            step_loss = self.fast_eval_step(state.params, batch=batch_TEST)
            test_loss += step_loss
            if self.interpolate and interpolate:
                ip_step_loss = self.fast_eval_step_interpol(state.params, batch=batch_TEST, interpol_kernels=interpol_kernels)
                interpolate_test_loss += ip_step_loss
        test_loss = test_loss/10
        interpolate_test_loss = interpolate_test_loss/10

        for _ in jnp.arange(num_batches_train):
            tr_rng, batch_rng = random.split(tr_rng, 2)
            batch,_ = self.data_generator.get_data(rng=batch_rng, batch_size=self.batch_size)
            state, _, _ = self.fast_train_step(state, batch=batch, interpol_call=False)

        if interpolate:  
            print(f'loss in epoch {epoch}: {test_loss},  interpolation loss: {interpolate_test_loss}')
        else:
            print(f'loss in epoch {epoch}: {test_loss}')
        return state, rng, test_loss, interpolate_test_loss, (listsnd, listmid, listlast)