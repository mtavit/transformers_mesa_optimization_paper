
'''Single-Task Data Generator for ICL Data.'''

import jax.numpy as jnp
from jax import random, lax, vmap, jit
from functools import partial
from typing import Tuple, Dict

from src.data.icl.icl_data import ICLDataGenerator


class ICLSingleGenerator(ICLDataGenerator):
    '''Concrete ICL Single-Task Data Generator Class.'''

    def __init__(self, 
                 seq_len:int, 
                 data_dim:int,
                 range:float, 
                 noise:float):
        '''
        Initializes the ICL Single Task Data Generator.
        Args:
            'seq_len' (int): The length of the sequence.
            'data_dim' (int): The dimension of the data.
            'range' (float): The range of the uniform distribution for the first token in a sequence U(-range, range).
            'noise' (float): The data noise level.
        '''
        super().__init__(noise=noise)
        self.seq_len = seq_len
        self.data_dim = data_dim
        self.range = range
        self.noise = noise

    def get_data(self, 
                 rng:random.PRNGKey, 
                 batch_size:int, 
                 **kwargs) -> Tuple[Tuple[jnp.ndarray]]:
        '''
        Generates a batch of data for regression task with & without eos-token.
        Args:
            'rng' (jax.random.PRNGKey): The random number generator key.
            'batch_size' (int): The size of the batch.
            Further args: 'use_prompt' (bool, Append prefix-prompt) and 'prompt' (jnp.ndarray, The prefix-prompt) and 'eos' (jnp.ndarray, The eos-token).
        Returns:
            tuple: A tuple containing two tuples of data and labels.
        '''
        return self.create_batch(rng=rng, 
                                 batch_size=batch_size, 
                                 data_dim=self.data_dim, 
                                 **kwargs)

    def get_data_info(self) -> Dict[str, any]:
        '''Returns datagenerator info as dict.'''
        return vars(self)

    @partial(jit, static_argnums=(0, 2, 3))
    def create_batch(self, 
                     rng: random.PRNGKey, 
                     batch_size: int, 
                     data_dim: int, 
                     **kwargs) -> Tuple[Tuple[jnp.ndarray]]:
        """
        Generates a batch of data for regression task with & without eos-token.
        Args:
            rng (jax.random.PRNGKey): The random number generator key.
            batch_size (int): The size of the batch.
            data_dim (int): The dimension of the data.
            kwargs: Further args, including use_prompt (bool, Append prefix-prompt) and prompt (jnp.ndarray, The prefix-prompt) and eos (jnp.ndarray, The eos-token).

        Returns:
            tuple: A tuple containing two or three tuples of data and labels, depenging on the "use_prompt" flag.
                If "use_prompt" is True, the tuple contains three tuples of data and labels:
                    - The first tuple contains the data and labels for the eos sequence.
                    - The second tuple contains the data and labels for the pure sequence.
                    - The third tuple contains the data and labels for the eos sequence with the prompt.
                Else:
                    - The first tuple contains the data and labels for the eos sequence.
                    - The second tuple contains the data and labels for the pure sequence.
        """

        use_prompt = kwargs.get('use_prompt', False)
        prompt = kwargs.get('prompt', None)
        eos = kwargs.get('eos', None)

        subkeyW, subkeyX = random.split(rng, 2)
        W = random.orthogonal(subkeyW, n=data_dim, shape=(batch_size,))
        X = random.uniform(subkeyX, shape=(batch_size, self.seq_len//3, data_dim), minval=-self.range, maxval=self.range)
        XR = random.uniform(subkeyX, shape=(batch_size, (self.seq_len//2 - self.seq_len//3), data_dim), minval=-self.range, maxval=self.range)
        XF = jnp.concatenate((X,XR),axis=1)

        data,labels = None,None
        if self.seq_len % 2 == 1:
            batch_seq = vmap(self.gen_one_seq, in_axes=(None,0,0,None))(rng,W,XF,self.seq_len//2)
            # TODO: The next line is just for fixing sizes, in experiments or finetuning, 
            #       data will be cut at seq_len//3 anyway, find cleaner solution here:
            batch_seq = jnp.concatenate((batch_seq, batch_seq[:,-2,:].reshape(batch_seq[:,-2,:].shape[0],1,batch_seq[:,-2,:].shape[-1])),axis=1)
            data, labels = batch_seq[:,:-1,:], batch_seq[:,1:,:]
        else:
            batch_seq = vmap(self.gen_one_seq, in_axes=(None,0,0,None))(rng,W,XF,self.seq_len//2)
            data, labels = batch_seq[:,:-1,:], batch_seq[:,1:,:]

        batch_seq_eos = vmap(self.gen_one_seq_eos, in_axes=(None,0,0,None,None))(rng,W,X,self.seq_len//3,eos)
        data_eos, labels_eos = batch_seq_eos[:,:-1,:], batch_seq_eos[:,1:,:]
        original_seq_len = data.shape[1]

        '''
        def _add_prompt(prompt, data):
            if prompt is None:
                return data
            return tuple([jnp.concatenate((jnp.broadcast_to(prompt,(batch_size, prompt.shape[0], prompt.shape[1])),content),axis=1)[:,:original_seq_len,:] for content in data])
        return lax.cond(use_prompt, ((data_eos,labels_eos), (data,labels), (_add_prompt(prompt, (data_eos,labels_eos)))),
                                    ((data_eos,labels_eos), (data,labels), (None, None)))
        '''
        
        # Convert prompt to a JAX array to handle None case
        prompt = jnp.zeros(shape=(0,data.shape[-1])) if prompt is None else prompt

        def _add_prompt(prompt, data):
            return tuple([jnp.concatenate((jnp.broadcast_to(prompt,(batch_size, prompt.shape[0], prompt.shape[1])),content),axis=1)[:,:original_seq_len,:] for content in data])

        def with_prompt(_):
            return ((data_eos, labels_eos), (data, labels), (_add_prompt(prompt, (data_eos, labels_eos))))

        def without_prompt(_):
            return ((data_eos, labels_eos), (data, labels), (jnp.zeros_like(data_eos), jnp.zeros_like(labels_eos)))

        return lax.cond(use_prompt, with_prompt, without_prompt, None)

    