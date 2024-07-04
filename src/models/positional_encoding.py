
'''Positional Encoding Module'''
import jax.numpy as jnp
import numpy as np
import math
from jax import vmap, device_put
from flax import linen as nn
from functools import partial

class PositionalEncoding(nn.Module):
    '''
    Class implementing the Positional Encoding for Transformer models.
    Fields:
        'pe_dim' (int): Dimension of the positional encoding
        'max_len' (int): Maximum length of input sequences
    '''
    pe_dim : int
    max_len : int = 10
    concat: bool = False

    def concat_single(self, x: jnp.ndarray, pe: jnp.ndarray) -> jnp.ndarray:
        '''Currently unused. Concatenates positional encoding to a single input tensor.'''
        return jnp.concatenate([x, pe], axis=-1)

    def concat_batch(self, x: jnp.ndarray, pe: jnp.ndarray) -> jnp.ndarray:
        '''Currently unused. Concatenates positional encoding to a batch of input tensors.'''
        myfunConcat = partial(self.concat_single, pe=pe)
        return vmap(myfunConcat)(x)

    def setup(self):
        '''Initializes the positional encoding.'''
        enc_size = self.pe_dim
        pe = np.zeros((self.max_len, enc_size))
        position = np.arange(0, self.max_len, dtype=np.float32)[:,None]
        div_term = np.exp(np.arange(0, enc_size, 2) * (-math.log(10000.0) / enc_size))
        pe[:, 0::2] = np.sin(position * div_term)
        if enc_size % 2 == 1:
            pe[:, 1::2] = np.cos(position * div_term)[:,:-1]
        else:
            pe[:, 1::2] = np.cos(position * div_term)
        self.pe = device_put(pe)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        '''Adds positional encoding to the input tensor.'''
        if self.concat:
            x = self.concat_batch(x, self.pe)
        else:
            x = x + self.pe
        return x