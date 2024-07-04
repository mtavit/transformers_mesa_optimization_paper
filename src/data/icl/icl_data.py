
'''In context learning data generator abstract class'''

import abc
import jax.numpy as jnp
from jax import random, vmap
from typing import Tuple, Dict

from src.data.datagenerator import DataGenerator

class ICLDataGenerator(DataGenerator):
    '''Abstract Base Class for ICL data generators'''
    def __init__(self, noise:float):
        self.noise = noise
        self.constr = False

    @abc.abstractmethod
    def get_data(self, 
                 rng:random.PRNGKey, 
                 batch_size:int, 
                 **kwargs) -> Tuple[Tuple[jnp.ndarray]]:
        '''Abstract method to get data batch'''
        raise NotImplementedError
        
    @abc.abstractmethod
    def get_data_info(self) -> Dict[str, any]:
        '''Abstract method to get data info'''
        raise NotImplementedError
    
    def _multi_mult(self, w:jnp.ndarray, X:jnp.ndarray) -> jnp.ndarray:
            '''
                Matrix multiplication for multiplication of every token in X with w from the left
                Args:
                    'w' (jnp.ndarray): weight matrix
                    'X' (jnp.ndarray): input matrix
                Returns:
                    result of multiplication of every token in X with w from the left
            '''
            return vmap(jnp.matmul, in_axes=(None,0))(w,X)
    
    def gen_one_seq_eos(self, 
                        rng:random.PRNGKey, 
                        w:jnp.ndarray, 
                        x:jnp.ndarray, 
                        sub_seq_length:int, 
                        eos:jnp.ndarray) -> jnp.ndarray:
        '''
            Generate a sequence with sub_seq_length*3 length with x, f_x and eos tokens
            Args:
                'rng' (random.PRNGKey): random key
                'w' (jnp.ndarray): weight matrix
                'x' (jnp.ndarray): input matrix
                'sub_seq_length' (int): length of the sequence
                'eos' (jnp.ndarray): end of sequence token
            Returns:
                sequence with sub_seq_length*3 length with x, f_x and eos tokens
        '''
        wx = self._multi_mult(w=w,X=x)
        f_x = wx + self.noise * random.normal(rng, shape=wx.shape)
        eos_tokens = jnp.ones(shape=(sub_seq_length,1)) @ (eos.T[None,...])
        result = jnp.zeros(shape=(sub_seq_length*3, x.shape[-1]))
        for i, update in enumerate([x, f_x, eos_tokens]):
            result = result.at[i::3, :].set(update)
        return result

    def gen_one_seq(self, 
                    rng:random.PRNGKey, 
                    w:jnp.ndarray, 
                    x:jnp.ndarray, 
                    sub_seq_length:int) -> jnp.ndarray:
        '''
            Generate a sequence with sub_seq_length*2 length with x and f_x tokens
            Args:
                rng: random key
                w: weight matrix
                x: input matrix
                sub_seq_length: length of the sequence
            Returns:
                sequence with sub_seq_length*2 length with x and f_x tokens
        '''
        wx = self._multi_mult(w=w,X=x)
        f_x = wx + self.noise * random.normal(rng, shape=wx.shape)
        result = jnp.zeros(shape=(sub_seq_length*2, x.shape[-1]))
        for i, update in enumerate([x, f_x]):
            result = result.at[i::2, :].set(update)
        return result
    
    @abc.abstractmethod
    def create_batch(self, 
                     rng:random.PRNGKey, 
                     batch_size:int, 
                     data_dim:int,
                     **kwargs) -> Tuple[Tuple[jnp.ndarray]]:
        '''Abstract method to create a batch of sequences'''
        raise NotImplementedError