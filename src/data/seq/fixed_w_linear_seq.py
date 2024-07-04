from jax import random, numpy as jnp, lax, vmap, jit
from functools import partial
from typing import Tuple, Dict

from src.data.seq.sequence_data import SequenceDataGenerator

class FixedWDataGenerator(SequenceDataGenerator):

    def __init__(self,
                 seq_len: int,
                 data_dim: int,
                 range: float,
                 noise: float,
                 noise_obs: float,
                 data_clip: float,
                 obs_dim: int = 10,
                 eye_obs: bool = True):
        super().__init__(seq_len=seq_len, data_dim=data_dim, eye_obs=eye_obs)
        self.obs_dim = obs_dim
        self.range = range
        self.noise = noise
        self.noise_obs = noise_obs
        self.data_clip = data_clip

    def get_data(self, 
                 rng:random.PRNGKey, 
                 batch_size:int) -> Tuple[Tuple[jnp.ndarray]]:
        '''
        Gets a batch of data with resp. partial observations.
        Args:
            'batch_size' (int): The batch size.
            'rng' (random.PRNGKey): The random number generator key.
        Returns:
            Tuple[Tuple[jnp.ndarray]]: A tuple of tuples containing the observed data and the original data in this order.
        '''
        return self.create_batch(rng=rng, 
                                 batch_size=batch_size, 
                                 data_dim=self.data_dim, 
                                 seq_len=self.seq_len,
                                 eye_obs=self.eye_obs)

    def get_data_info(self) -> Dict[str, any]:
        '''Returns the data information as a dict.'''
        return vars(self)

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
            next_x = jnp.matmul(W, prev_x) + self.noise * random.normal(rng, shape=x_1.shape)
            next_x = jnp.clip(next_x, -self.data_clip, self.data_clip)
            return next_x, next_x
        _, sequence = lax.scan(step, x_1, seq_rng[:-1])
        sequence = jnp.concatenate([jnp.expand_dims(x_1, 0), sequence], axis=0)
        return sequence
    
    def _obs_and_noise(self, 
                       obs_mat: jnp.ndarray,
                       x: jnp.ndarray, 
                       noise: jnp.ndarray) -> jnp.ndarray:
        '''
        Applies a linear transformation to the input matrix `mat` and vector `x`,
        and adds noise to the result.
        Parameters:
            'obs_mat' (ndarray): The observation matrix [B, obs_dim, data_dim]
            'x' (ndarray): The hidden states [B, seq_len, data_dim]
            'noise' (ndarray): The noise vector

        Returns:
            ndarray: Observed data with added gaussian noise
        '''
        return vmap(vmap(jnp.matmul,in_axes=(None,0)), in_axes=(0,0))(obs_mat, x) + noise

    @partial(jit, static_argnums=(0,2,3,4,5))
    def create_batch(self, 
                     rng: random.PRNGKey, 
                     batch_size: int, 
                     data_dim: int, 
                     seq_len: int,
                     eye_obs: bool) -> Tuple[Tuple[jnp.ndarray]]:
        '''
        Creates a batch of linear sequences
        Args:
            'rng' (PRNGKey): The random number generator key
            'batch_size' (int): The batch size
            'data_dim' (int): The dimensionality of the data
            'seq_len' (int): The length of the sequence
            'eye_obs' (bool): Use raw hidden states as observations/inputs to model
        Returns:
            Tuple[ndarray]: The batch of observed data and the batch of original data
        '''
        rng, _, subkeyX, subkeyN1, subkeyN2, subkeyObs = random.split(rng, 6)
        batch_of_noise_keys = random.split(subkeyN1, batch_size)
        W = random.orthogonal(key=random.PRNGKey(42), 
                              n=data_dim)
        W = jnp.broadcast_to(W, (batch_size,) + W.shape)
        
        X = random.uniform(key=subkeyX, 
                           shape=(batch_size, data_dim), 
                           minval=-self.range, 
                           maxval=self.range)
        
        dataset = vmap(partial(self.generate_sequence, seq_length=seq_len+1))(W=W,x_1=X,rng=batch_of_noise_keys)
        original_data, original_labels = dataset[:,:-1,:], dataset[:,1:,:]

        obs_mat = jnp.eye(self.obs_dim)[None, :, :].repeat(batch_size, axis=0) if eye_obs else 0.5*random.normal(subkeyObs, shape=(batch_size,self.obs_dim,data_dim))

        new_noise = self.noise_obs * random.normal(subkeyN2, shape=(original_data.shape[0],original_data.shape[1]+1,self.obs_dim))
        observed_data = self._obs_and_noise(obs_mat, original_data, new_noise[:,:-1,:])
        observed_labels = self._obs_and_noise(obs_mat, original_labels, new_noise[:,1:,:])

        return (observed_data, observed_labels), (original_data, original_labels)