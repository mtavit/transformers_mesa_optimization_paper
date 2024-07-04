
'''Token-Finetuner Abstract Base Class'''

import abc
from jax import random, numpy as jnp
from flax import linen as nn
from flax.training import train_state

from typing import Dict, Any

from src.data.datagenerator import DataGenerator

class FineTuner(metaclass=abc.ABCMeta):
    '''Abstract Base Class for DataGenerator'''
    def __init__(self, 
                 model_tf: nn.Module, 
                 state_tf: train_state.TrainState, 
                 data_generator: DataGenerator,
                 train_batch_size: int,
                 finetune_steps: int):
        '''
        Initializes the Token-Finetuner.
        Args:
            'model_tf' (nn.Module): The Flax (Transformer) Model.
            'state_tf' (train_state.TrainState): The Flax TrainState for the model.
            'data_generator' (DataGenerator): The DataGenerator.
            'train_batch_size' (int): The batch size for finetuning.
            'finetune_steps' (int): The number of finetuning steps.
        '''
        
        self.model_tf = model_tf
        self.state_tf = state_tf
        self.data_generator = data_generator
        self.train_batch_size = train_batch_size
        self.finetune_steps = finetune_steps

    @abc.abstractmethod
    def finetune(self, 
                 rng: random.PRNGKey, 
                 finetune_lr: float) -> jnp.ndarray:
        '''Abstract method for finetuning'''
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_tf_pred(self, 
                    rng: random.PRNGKey, 
                    tf_params: Dict[str, Any], 
                    **kwargs) -> jnp.ndarray:
        '''Abstract method for executing tf with current finetuned data'''
        raise NotImplementedError
    
    @abc.abstractmethod
    def loss_fn(self, 
                rng: random.PRNGKey, 
                tf_params: Dict[str, Any], 
                finetune_params: Dict[str, Any]) -> float:
        '''Abstract method for finetuning loss'''
        raise NotImplementedError
    
    @abc.abstractmethod
    def train_step_ft(self, 
                      rng: random.PRNGKey,
                      state_finetune: train_state.TrainState, 
                      state_tf: train_state.TrainState) -> train_state.TrainState:
        '''Abstract method for finetuning step'''
        raise NotImplementedError