from jax import random, lax, nn, numpy as jnp, vmap, jit
from functools import partial
from typing import Tuple, Dict

from src.data.seq.sequence_data import SequenceDataGenerator

class NonlinearSequenceDataGenerator(SequenceDataGenerator):

    def __init__(self, 
                 A_mat:jnp.ndarray,
                 B_mat:jnp.ndarray,
                 seq_len:int, 
                 data_dim:int,
                 range:float,
                 noise:float):
        super().__init__(seq_len=seq_len, data_dim=data_dim, eye_obs=True)
        self.A_mat = A_mat
        self.B_mat = B_mat
        self.seq_len = seq_len
        self.data_dim = data_dim
        self.range = range
        self.noise = noise
        self.obs_dim = data_dim

    def get_data(self,  
                 rng:random.PRNGKey,
                 batch_size:int) -> Tuple[Tuple[jnp.ndarray]]:
        '''
        Gets a batch of data.
        Args:
            'batch_size' (int): The batch size.
            'rng' (random.PRNGKey): The random number generator key.
        Returns:
            Tuple[Tuple[jnp.ndarray]]: A tuple of tuples containing the nonlinear data and labels
        '''
        return self.create_batch(rng=rng, 
                                 batch_size=batch_size, 
                                 data_dim=self.data_dim, 
                                 seq_len=self.seq_len)
    
    def get_data_info(self) -> Dict:
        '''Returns the data information as a dict.'''
        return vars(self)
    
    def _mini_mlp(self, x: jnp.ndarray) -> jnp.ndarray:
        '''Implementation of MLP-based nonlinearity.'''
        return (nn.gelu(x@self.A_mat)@self.B_mat)

    def generate_sequence(self, 
                          W: jnp.ndarray, 
                          x_1: jnp.ndarray, 
                          seq_length: int, 
                          rng: random.PRNGKey) -> jnp.ndarray:
        '''
        Generates a sequence of tokens
        Args:
            'W' (ndarray): The weight matrix [D, D]
            'x_1' (ndarray): The initial input vector [D]
            'seq_length' (int): The length S of the sequence
            'rng' (PRNGKey): The random number generator key for added gaussian noise
        Returns:
            'ndarray': The generated sequence [S, D]
        '''
        seq_rng  = random.split(rng, seq_length)
        def step(prev_x, rng):
            f_x = self._mini_mlp(prev_x)
            f_x_normed = f_x/(jnp.linalg.norm(f_x)+1e-16)
            W_x = jnp.matmul(W, f_x_normed)
            next_x = W_x + self.noise * random.normal(rng, shape=(x_1.shape))
            return next_x, next_x
        _, sequence = lax.scan(step, x_1, seq_rng[:-1])
        sequence = jnp.concatenate([jnp.expand_dims(x_1, 0), sequence], axis=0)
        return sequence

    @partial(jit, static_argnums=(0, 2, 3, 4))
    def create_batch(self, 
                     rng: random.PRNGKey, 
                     batch_size: int, 
                     data_dim: int, 
                     seq_len: int) -> Tuple[Tuple[jnp.ndarray]]:
        '''
        Creates a batch of nonlinear sequence data.
        Args:
            'rng' (PRNGKey): The random number generator key.
            'batch_size' (int): The batch size.
            'data_dim' (int): The data dimension.
            'seq_len' (int): The sequence length.
        Returns:
            Tuple[Tuple[jnp.ndarray]]: The batch of data and labels.
        '''
        rng, subkeyW, subkeyX, subkeyN  = random.split(rng, 4)
        batch_of_noise_keys = random.split(subkeyN, batch_size)

        W = random.orthogonal(subkeyW, n=data_dim, shape=(batch_size,))
        X = random.uniform(subkeyX, shape=(batch_size, data_dim), minval=-self.range, maxval=self.range)

        dataset = vmap(partial(self.generate_sequence, seq_length=seq_len+1))(W=W,x_1=X,rng=batch_of_noise_keys)
        data, labels = dataset[:,:-1,:], dataset[:,1:,:]
        return (data,labels), None