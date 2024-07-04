import abc
from jax import random, numpy as jnp
from typing import Tuple, Dict

from src.data.datagenerator import DataGenerator

class SequenceDataGenerator(DataGenerator):

    def __init__(self, seq_len: int, data_dim: int, eye_obs: bool):
        self.seq_len = seq_len
        self.data_dim = data_dim
        self.constr = False
        self.eye_obs = eye_obs

    @abc.abstractmethod
    def get_data(self, 
                 rng: random.PRNGKey, 
                 batch_size: int) -> Tuple[Tuple[jnp.ndarray]]:
        '''Abstract method to get data batch'''
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_data_info(self) -> Dict[str, any]:
        '''Abstract method to get data info'''
        raise NotImplementedError

    @abc.abstractmethod
    def generate_sequence(self, 
                          W: jnp.ndarray, 
                          x_1: jnp.ndarray, 
                          seq_length: int, 
                          rng: random.PRNGKey) -> jnp.ndarray:
        '''Abstract method to generate a sequence'''
        raise NotImplementedError

    @abc.abstractmethod
    def create_batch(self, 
                     rng: random.PRNGKey, 
                     batch_size: int, 
                     data_dim: int, 
                     seq_len: int) -> Tuple[Tuple[jnp.ndarray]]:
        '''Abstract method to create a batch of sequences'''
        raise NotImplementedError