
'''Constructed data generator classes'''

from jax import random, numpy as jnp, jit
from functools import partial
from typing import Tuple, Dict

from src.data.datagenerator import DataGenerator

class ConstructedPartObsGenerator(DataGenerator):
    '''Data generator for constructing data with partial observations'''
    def __init__(self, 
                 data_generator: DataGenerator,
                 embed_dim: int):
        '''
        Initializes the ConstructedPartObsGenerator.
        Args:
            'data_generator' (DataGenerator): The data generator.
            'embed_dim' (int): The embedding dimension.
        '''
        
        self.data_generator = data_generator
        self.embed_dim = embed_dim
        self.seq_len = data_generator.get_data_info()['seq_len']
        self.data_dim = data_generator.get_data_info()['data_dim']
        self.obs_dim = data_generator.get_data_info()['obs_dim']
        self.constr = True
        self.slots = self.embed_dim // self.obs_dim

    def get_data(self, 
                 rng: random.PRNGKey, 
                 batch_size: int) -> Tuple[Tuple[jnp.ndarray]]:
        '''
        Gets a batch of constructed data with partial observations.
        Args:
            'rng' (random.PRNGKey): The random number generator key.
            'batch_size' (int): The batch size.
        Returns:
            Tuple[Tuple[jnp.ndarray]]: A tuple containing the constructed data and the original data.
        '''
        return self.create_batch(rng=rng, 
                                 batch_size=batch_size)

    def get_data_info(self) -> Dict[str, any]:
        '''Returns the data information as a dict.'''
        return vars(self)

    @partial(jit, static_argnums=(0,2))
    def create_batch(self, 
                     rng: random.PRNGKey, 
                     batch_size: int) -> Tuple[Tuple[jnp.ndarray]]:
        '''
        Creates a batch of constructed data with partial observations.
        Args:
            'rng' (random.PRNGKey): The random number generator key.
            'batch_size' (int): The batch size.
        Returns:
            Tuple[Tuple[jnp.ndarray]]: A tuple containing the constructed data and the original data.
        '''
        (observed_data, observed_labels), (original_data, original_labels) = self.data_generator.get_data(rng=rng, batch_size=batch_size)
        constructed_data = jnp.zeros(shape=(batch_size, self.seq_len, self.embed_dim))
        constructed_data = constructed_data.at[:,:,0:self.obs_dim].set(observed_data)
        for k in range(1, self.embed_dim // self.obs_dim):
            shifted_data = jnp.concatenate((jnp.zeros(shape=(batch_size,(k),self.obs_dim)),observed_data[:,:-1*(k),:]),axis=1)
            constructed_data = constructed_data.at[:,:,k*self.obs_dim:(k+1)*self.obs_dim].set(shifted_data)
        return (constructed_data, observed_labels), (original_data, original_labels)


class ConstructedFullSeqGenerator(DataGenerator):
    '''Data generator for constructing data with fully observed sequences'''
    def __init__(self, 
                 data_generator: DataGenerator,
                 embed_dim: int):
        '''
        Initializes the ConstructedFullSeqGenerator.
        Args:
            'data_generator' (DataGenerator): The data generator.
            'embed_dim' (int): The embedding dimension.
        '''
        self.data_generator = data_generator
        self.embed_dim = embed_dim
        self.seq_len = data_generator.get_data_info()['seq_len']
        self.data_dim = data_generator.get_data_info()['data_dim']
        self.constr = True
        self.slots = 4 # Fixed
        self.obs_dim = data_generator.get_data_info()['obs_dim']

    @partial(jit, static_argnums=(0,2))
    def get_data(self, 
                 rng: random.PRNGKey, 
                 batch_size: int):
        '''
        Gets a batch of constructed data with fully observed sequences.
        Args:
            'rng' (random.PRNGKey): The random number generator key.
            'batch_size' (int): The batch size.
        Returns:
            Tuple[Tuple[jnp.ndarray]]: A tuple containing the constructed data and the original data.
        '''
        return self.create_batch(rng=rng, 
                                 batch_size=batch_size)

    def get_data_info(self) -> Dict[str, any]:
        '''
        Gets the data information.
        Returns:
            Dict[str, any]: The data information.
        '''
        return vars(self)

    # TODO: Generalize this to any number of slots
    @partial(jit, static_argnums=(0,2))
    def create_batch(self, 
                     rng: random.PRNGKey, 
                     batch_size: int) -> Tuple[Tuple[jnp.ndarray]]:
        '''
        Creates a batch of constructed ([0,x_i,x_i,x_{i-1}]) data with fully observed sequences.
        Args:
            'rng' (random.PRNGKey): The random number generator key.
            'batch_size' (int): The batch size.
        Returns:
            Tuple[Tuple[jnp.ndarray]]: A tuple containing the constructed data and the original data.
        '''
        (observed_data, observed_labels), (original_data, original_labels) = self.data_generator.get_data(rng=rng, batch_size=batch_size)
        constructed_data = jnp.zeros(shape=(batch_size, self.seq_len, self.embed_dim))
        constructed_data = constructed_data.at[:,:,self.obs_dim:2*self.obs_dim].set(observed_data)
        constructed_data = constructed_data.at[:,:,2*self.obs_dim:3*self.obs_dim].set(observed_data)
        shifted_data = jnp.concatenate((jnp.zeros(shape=(batch_size,1,self.obs_dim)),observed_data[:,:-1,:]),axis=1)
        constructed_data = constructed_data.at[:,:,3*self.obs_dim:].set(shifted_data)
        return (constructed_data, observed_labels), (original_data, original_labels)