
'''Multi-Task Data Generator for ICL Data.'''

import jax.numpy as jnp
from jax import random, jit
from typing import Tuple, Dict
from functools import partial
from src.data.icl.icl_data import ICLDataGenerator
from src.data.icl.icl_single import ICLSingleGenerator

class ICLMultiGenerator(ICLDataGenerator):
    '''Concrete ICL Multi-Task Data Generator Class.'''

    def __init__(self, 
                 seq_len_task1:int, 
                 seq_len_task2:int,
                 data_dim:int,
                 range:float,
                 noise:float):
        '''
        Initializes the ICL Multi Task Data Generator.
        Args:
            'seq_len_task1' (int): The length of the first sequence.
            'seq_len_task2' (int): The length of the second sequence.
            'data_dim' (int): The dimension of the data.
            'range' (float): The range of the uniform distribution for the first token in a sequence U(-range, range).
            'noise' (float): The noise level.
        '''
        
        super().__init__(noise=noise)
        self.seq_len_task1 = seq_len_task1
        self.seq_len_task2 = seq_len_task2
        self.data_dim = data_dim
        self.range = range       
        self.noise = noise

        self.task1_generator = self._get_icl_single(self.seq_len_task1)
        self.task2_generator = self._get_icl_single(self.seq_len_task2)
    
    def _get_icl_single(self, seq_len:int) -> ICLSingleGenerator:
        '''
        Initializes the ICL Single Task Data Generators.
        Args:
            'seq_len' (int): The length of the sequence.
        Returns:
            ICLSingle: The ICL Single Task Data Generator.
        '''
        return ICLSingleGenerator(seq_len=seq_len,
                                    data_dim=self.data_dim,
                                    range=self.range,
                                    noise=self.noise)

    def get_data(self, 
                 rng:random.PRNGKey,  
                 batch_size:int, 
                 **kwargs) -> Tuple[Tuple[jnp.ndarray]]:
        '''
        Generates a batch of data for multitask regression task with & without eos-token.
        Args:
            'rng' (jax.random.PRNGKey): The random number generator key.
            'batch_size' (int): The size of the batch.
            'kwargs': Includes:
                - 'prompt' (jnp.ndarray): prefix-prompt
                - 'eos_token' (jnp.ndarray): end of sentence token
        Returns:
            tuple: A tuple containing three tuples of data and labels.
        '''
        return self.create_batch(rng=rng, 
                                 batch_size=batch_size,
                                 data_dim=self.data_dim, 
                                 **kwargs)

    def get_data_info(self) -> Dict[str, any]:
        '''returns datagenerator info as dict'''
        return vars(self)
    
    @partial(jit, static_argnums=(0, 2))
    def create_batch(self, 
                     rng: random.PRNGKey, 
                     batch_size: int,
                     **kwargs) -> Tuple[Tuple[jnp.ndarray]]:
        '''
        Generates a batch of data for multitask regression task with & without eos-token.
        Args:
            rng (jax.random.PRNGKey): The random number generator key.
            'batch_size' (int): The size of the batch.
            'data_dim' (int): The dimension of the data.
            'kwargs': Includes:
                - 'prompt' (jnp.ndarray): prefix-prompt
                - 'eos_token' (jnp.ndarray): end of sentence token
        Returns:
            tuple: A tuple containing three tuples of data and labels.
                - The first tuple contains the data and labels for the eos sequence.
                - The second tuple contains the data and labels for the pure sequence.
                - The third tuple contains the data and labels for the eos+prompt sequence.
        '''
        prompt = kwargs.get('prompt', None)
        eos = kwargs.get('eos', None)

        rng, sX1, sX2 = random.split(rng, 3)
        
        data_task1 = self.task1_generator.get_data(rng=sX1, batch_size=batch_size, use_prompt=True, prompt=prompt, eos=eos)
        (data_eos1,labels_eos1), (data1,labels1), (data_eos_p1, label_eos_p1) = data_task1

        data_task2 = self.task2_generator.get_data(rng=sX2, batch_size=batch_size, use_prompt=True, prompt=prompt, eos=eos)
        (data_eos2,labels_eos2), (data2,labels2), (data_eos_p2, label_eos_p2) = data_task2
        

        batch = jnp.concatenate((data1[:,:,:], data2), axis=1), jnp.concatenate((labels1[:,:,:], labels2), axis=1)
        eos_batch = jnp.concatenate((data_eos1[:,:,:], data_eos2), axis=1), jnp.concatenate((labels_eos1[:,:,:], labels_eos2), axis=1)
        p_eos_batch = jnp.concatenate((data_eos_p1[:,:,:], data_eos_p2), axis=1), jnp.concatenate((label_eos_p1[:,:,:], label_eos_p2), axis=1)  

        return eos_batch, batch, p_eos_batch