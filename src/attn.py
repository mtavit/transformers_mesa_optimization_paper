
'''Module for Multi-Head Attention Layer'''

import flax.linen as nn
import jax.numpy as jnp
import math
from typing import Callable
from jax import random

from src.models.positional_encoding import PositionalEncoding

class MultiHeadAttention(nn.Module):
    '''
    Multi-Head Attention Layer 
    Fields:
        'masked' (bool): Flag whether to use masked attention
        'embed_dim' (int): Embedding dimension
        'num_heads' (int): Number of heads
        'use_softmax' (bool): Flag whether to use softmax
        'use_bias' (bool): Flag whether to use bias
        'key_size' (int): Key size
        'interpol' (bool): Flag whether to use interpolation
        'initializer' (Callable[[random.PRNGKey, tuple, jnp.dtype], jnp.ndarray]): Initializer function
        'seq_len' (int): Sequence length
        'use_pe_kq' (bool): Flag whether to use (special) positional encoding in the query and key vectors
    '''
    masked : bool
    embed_dim : int
    num_heads : int
    use_softmax : bool
    use_bias : bool
    key_size : int
    interpol : bool
    initializer : Callable[[random.PRNGKey, tuple, jnp.dtype], jnp.ndarray]
    seq_len : int
    use_pe_kq : bool = False
    use_schlagnorm : bool = False
    schlagnorm_targets : bool = False

    def scaled_dot_product_attention(self,
                                     q: jnp.ndarray,
                                     k: jnp.ndarray,
                                     v: jnp.ndarray,
                                     use_softmax: bool,
                                     masked=True) -> jnp.ndarray:
        '''
        Scaled Dot-Product Attention
        Args:
            'q' (jnp.ndarray): Query projections
            'k' (jnp.ndarray): Key projections
            'v' (jnp.ndarray): Value projections
            'use_softmax' (bool): Flag whether to use softmax-attention
            'masked' (bool): Flag whether to use causally masked attention
        Returns:
            Tuple[jnp.ndarray]: Attention values and attention logits.
        '''
        d_k = q.shape[-1]                                            # get dimension of q,k,v
        attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))         # compute inner product
        if masked:
            # Autoregressive mask
            mask = jnp.tril(jnp.ones(shape=(attn_logits.shape)), k=0)     # Creates Lower Tri-Mat. with 0s everywhere else
            attn_logits = jnp.where(mask == 0, 0, attn_logits)          # add mask (for Linear)
            if use_softmax:
                attn_logits = jnp.where(mask == 0, -1e30, attn_logits/math.sqrt(d_k))   # add mask (for SoftMax)
        if use_softmax:
            attn_logits = nn.softmax(attn_logits, axis=-1)
        values = jnp.matmul(attn_logits, v)                          # matrix product with Values
        return values, attn_logits

    def setup(self):
        '''Initializes the Multi-Head Attention Layer with the specified parameters.'''
        def create_dense_layer(output_dim: int) -> nn.Module:
            '''
            Auxiliary function to create a dense layer with the specified output dimension,
            using the class initializer and bias setting.
            Args:
                output_dim (int): Output dimension of the created layer
            '''
            return nn.Dense(features=output_dim,
                            kernel_init=self.initializer,
                            use_bias=self.use_bias)

        # Layer dimensions
        proj_dim = self.key_size * self.num_heads

        # Layer specifications
        layer_specs = {
            'q_proj': proj_dim, 'v_proj': proj_dim, 'k_proj': proj_dim, 'o_proj': self.embed_dim,
            'q_fm': proj_dim, 'v_fm': proj_dim, 'k_fm': proj_dim, 'o_fm': self.embed_dim
        }

        # Create Layers
        for layer_name, output_dim in layer_specs.items():
            setattr(self, layer_name, create_dense_layer(output_dim))
        
        # Optional: Positional Encoding only at K,Q:
        if self.use_pe_kq:
            self.pos_enc_kq = PositionalEncoding(pe_dim = self.embed_dim, max_len=self.seq_len, concat=True)
    
    def __call__(self, x: jnp.ndarray, interpol_call: bool=False) -> jnp.ndarray:
        '''Applies the Multi-Head Attention Layer to the input tensor.'''
        def helper_func_seperate_q_k_v(q,k,v):
            q = q.reshape(batch_size, seq_length, self.num_heads, self.key_size).transpose((0, 2, 1, 3))
            k = k.reshape(batch_size, seq_length, self.num_heads, self.key_size).transpose((0, 2, 1, 3))
            v = v.reshape(batch_size, seq_length, self.num_heads, self.key_size).transpose((0, 2, 1, 3))
            return q,k,v

        batch_size, seq_length, _ = x.shape

        
        # Optional: Positional Encoding only at K,Q in first layer:
        if self.use_pe_kq:
            t = self.pos_enc_kq(x)
            k = self.k_proj(t)
            q = self.q_proj(t)
        else:
            q = self.q_proj(x)
            k = self.k_proj(x)  
        v = self.v_proj(x)

        # Separate Q, K, V from linear output
        q,k,v = helper_func_seperate_q_k_v(q,k,v)

        # Optional: Normalize using SchlagNorm
        if self.use_schlagnorm:
            q = q/(1e-16+jnp.linalg.norm(q, axis=-1)[...,None])
            k = k/(1e-16+jnp.linalg.norm(k, axis=-1)[...,None])
            if self.schlagnorm_targets:
                v = v/(1e-16+jnp.linalg.norm(v, axis=-1)[...,None])

        # Optional: Interpolation
        if interpol_call:
            d_k = q.shape[-1]
            q_ip = self.q_fm(x)
            k_ip = self.k_fm(x)
            v_ip = self.v_fm(x)
            q_ip,k_ip,v_ip = helper_func_seperate_q_k_v(q_ip,k_ip,v_ip)
            
            attn_logits_comb = 0.5*(jnp.matmul(q, jnp.swapaxes(k, -2, -1)) + jnp.matmul(q_ip, jnp.swapaxes(k_ip, -2, -1)))

            mask = jnp.tril(jnp.ones(shape=(attn_logits_comb.shape)), k=0)
            if self.use_softmax:
                attn_logits_comb = jnp.where(mask == 0, -1e30, attn_logits_comb/math.sqrt(d_k))
                attn_logits_comb = nn.softmax(attn_logits_comb, axis=-1)
            else:
                attn_logits_comb = jnp.where(mask == 0, 0, attn_logits_comb)

            res1 = attn_logits_comb@v
            res2 = attn_logits_comb@v_ip
            res1p = res1.transpose((0, 2, 1, 3)).reshape(batch_size, seq_length, self.key_size*self.num_heads)
            res2p = res2.transpose((0, 2, 1, 3)).reshape(batch_size, seq_length, self.key_size*self.num_heads)
            res_o = 0.5*(self.o_proj(res1p) + self.o_fm(res2p))
            return (res_o), attn_logits_comb, None
        
        else:
            values, attn_logits = self.scaled_dot_product_attention(q=q, k=k, v=v, masked=self.masked, use_softmax=self.use_softmax)
            values = values.transpose((0, 2, 1, 3)).reshape(batch_size, seq_length, self.key_size*self.num_heads)
            o = self.o_proj(values)
            return o, attn_logits, (q,k,v)