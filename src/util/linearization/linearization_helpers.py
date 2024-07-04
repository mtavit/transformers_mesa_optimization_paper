
'''Module for linearization helper class.'''

import flax
import jax.numpy as jnp
from jax import jit, value_and_grad, random, device_get
from flax import linen as nn
from flax.training import train_state
from functools import partial
from typing import Tuple

from src.data.seq.sequence_data import SequenceDataGenerator
from src.train import _compute_loss


class LinearizationHelper:
    '''Helper class for linearization experiments.'''
    def __init__(self,
                 params_softmax : flax.core.frozen_dict.FrozenDict,
                 model_tf : nn.Module,
                 linear_layer : nn.Module,
                 data_generator : SequenceDataGenerator,
                 layer_idx: int,
                 num_batches_train: int,
                 linearization_batch_size: int = 64):
        '''
        Initializes the LinearizationHelper.
        Args:
            'params_softmax' (flax.core.frozen_dict.FrozenDict): The parameters of the softmax model.
            'model_tf' (nn.Module): The flax model.
            'linear_layer' (nn.Module): The linear layer.
            'data_generator' (DataGenerator): The data generator.
            'layer_idx' (int): The index of the layer to linearize.
            'num_batches_train' (int): The number of batches to train the linear layer on.
        '''
        self.params_softmax = params_softmax
        self.model_tf = model_tf
        self.linear_layer = linear_layer
        self.data_generator = data_generator
        self.layer_idx = layer_idx
        self.num_batches_train = num_batches_train
        self.linearization_batch_size = linearization_batch_size
    
    @partial(jit, static_argnums=(0))
    def create_data(self, rng: random.PRNGKey) -> Tuple[jnp.ndarray]:
        '''
        Create data for linearization.
        Args:
            'rng' (random.PRNGKey): The random number generator.
        Returns:
            'activations' (jnp.ndarray): The activations of the specified layer.
            'attention_outputs' (jnp.ndarray): The attention outputs of the specified layer.
        '''
        (batch_data, _), _ = self.data_generator.get_data(rng=rng, batch_size=self.linearization_batch_size)
        _, (activations, _, attention_outputs, _) = self.model_tf.apply({'params': self.params_softmax}, batch_data)
        return activations[self.layer_idx], attention_outputs[self.layer_idx]

    def calculate_loss_lin(self, 
                           params: flax.core.frozen_dict.FrozenDict, 
                           batch: Tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        '''Computes loss for linearization.'''
        inp_data, labels = batch
        preds, _, _ = self.linear_layer.apply({'params': params}, inp_data)
        loss = _compute_loss(preds=preds, targets=labels)
        return loss

    @partial(jit, static_argnums=(0))
    def train_step_lin(self, 
                       state: train_state.TrainState, 
                       batch: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[train_state.TrainState, jnp.ndarray]:
        '''
        Perform a training step for linearization.
        Args:
            'state' (train_state.TrainState): The current state.
            'batch' (Tuple[jnp.ndarray, jnp.ndarray]): The input batch.
        Returns: Tuple[train_state.TrainState, jnp.ndarray]:
            'state' (train_state.TrainState): The updated state.
            'loss' (jnp.ndarray): The calculated loss.
        '''
        loss_fn = lambda params: self.calculate_loss_lin(params=params, batch=batch)
        loss, grads = value_and_grad(loss_fn, has_aux=False)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    @partial(jit, static_argnums=(0))
    def eval_step_lin(self, 
                      params: flax.core.frozen_dict.FrozenDict, 
                      batch: Tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        '''
        Perform an evaluation step for linearization.
        Args:
            'params' (flax.core.frozen_dict.FrozenDict): The parameters.
            'batch' (Tuple[jnp.ndarray, jnp.ndarray]): The input batch.
        Returns:
            jnp.ndarray: The calculated loss.
        '''
        return self.calculate_loss_lin(params=params, batch=batch)

    def train_epoch_lin(self, 
                        rng: random.PRNGKey, 
                        state: train_state.TrainState, 
                        test_rng: random.PRNGKey) -> Tuple[train_state.TrainState, random.PRNGKey, jnp.ndarray]:
        '''
        Perform a training epoch for linearization.
        Args:
            'rng' (random.PRNGKey): The random number generator.
            'state' (train_state.TrainState): The current state for the linear layer.
            'test_rng' (random.PRNGKey): The random number generator for testing.
        Returns:
            Tuple[train_state.TrainState, random.PRNGKey, jnp.ndarray]:
                - The updated state for the linear layer.
                - The updated random number generator.
                - The average test loss.
        '''
        rng, tr_rng = random.split(rng)
        test_loss = 0
        for _ in range(10):
            test_rng, batch_rng = random.split(test_rng, 2)
            batch_test = self.create_data(batch_rng)
            test_loss += self.eval_step_lin(params=state.params, batch=batch_test)
        test_loss = test_loss/10
        for _ in range(self.num_batches_train):
            tr_rng, batch_rng = random.split(tr_rng, 2)
            batch = self.create_data(batch_rng)
            state, _ = self.train_step_lin(state=state, batch=batch) 
        return state, rng, test_loss