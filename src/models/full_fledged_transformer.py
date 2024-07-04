
'''Module for the full-fledged Transformer model. Contains the (single) TransformerBlock, TF_BlockList, and FullTransformerModel classes.'''

import flax.linen as nn
import jax.numpy as jnp
from jax import random
from jax import nn as jax_net
from typing import Callable

from src.attn import MultiHeadAttention
from src.mesa import MultiHeadMesa
from src.models.positional_encoding import PositionalEncoding

class TransformerBlock(nn.Module):
    '''
    Single Transformer Block Can be used with or without MESA or MLP or LayerNorms. 
    Fields:
        'data_dim' (int): Dimension of the (original) data tokens (for masking purposes in construction experiments)
        'embed_dim' (int): Dimension of the embeddings
        'key_size' (int): Size of the key vectors
        'num_heads' (int): Number of attention heads
        'dim_feedforward' (int): Dimension of the feedforward network
        'use_layer_norm' (bool): Whether to use LayerNorm
        'use_bias' (bool): Flag to use bias in the attention layer
        'use_softmax' (bool): Flag to use softmax in the attention layer
        'use_mlp' (bool): Flag to use the MLP
        'masked' (bool): Flag to use masking in the attention layer
        'interpol' (bool): Flag to use interpolation in the attention layer
        'mask_inputs' (bool): Flag to mask the input data
        'use_mesa' (bool): Flag to use MESA
        'standard_backprop_mesa' (bool): Flag to use standard backpropagation in MESA
        'normalization_mesa' (bool): Flag to use normalization in MESA
        'initializer' (Callable[[random.PRNGKey, tuple, jnp.dtype], jnp.ndarray]): Initializer for the weights
        'use_pe_kq' (bool): Flag to use positional encoding in the query and key vectors
        'seq_len' (int): Length of the input sequence
        'use_schlagnorm' (bool): Flag to use Schlag-Norm (used in nonlinear experiments)
        'schlagnorm_targets' (bool): Flag to use Schlag-Norm on the targets (used in nonlinear experiments)
    '''
    data_dim : int
    embed_dim : int
    key_size : int
    num_heads : int
    dim_feedforward : int
    use_layer_norm : bool
    use_bias : bool
    use_softmax : bool
    use_mlp : bool
    masked : bool
    interpol : bool
    mask_inputs : bool
    use_mesa : bool
    standard_backprop_mesa : bool
    normalization_mesa : bool
    initializer : Callable[[random.PRNGKey, tuple, jnp.dtype], jnp.ndarray]
    use_pe_kq : bool
    seq_len: int
    use_schlagnorm : bool
    schlagnorm_targets : bool

    def setup(self):
        '''Initializes the Transformer Block.'''
        # Attention Layer
        if self.use_mesa:
            self.self_attn = MultiHeadMesa(num_heads=self.num_heads, 
                                           input_size=self.key_size, 
                                           emb_size=self.embed_dim,
                                           seq_len=self.seq_len, 
                                           use_bias_p=self.use_bias, 
                                           standard_backprop=self.standard_backprop_mesa, 
                                           normalization=self.normalization_mesa, 
                                           initializer=self.initializer,
                                           use_schlagnorm=self.use_schlagnorm,
                                           schlagnorm_targets=self.schlagnorm_targets,
                                           use_pe_kq=self.use_pe_kq)
        else:
            self.self_attn = MultiHeadAttention(interpol=self.interpol, 
                                                num_heads=self.num_heads, 
                                                embed_dim=self.embed_dim, 
                                                masked=self.masked, 
                                                use_softmax=self.use_softmax, 
                                                use_bias=self.use_bias, 
                                                key_size=self.key_size, 
                                                initializer=self.initializer, 
                                                use_pe_kq=self.use_pe_kq, 
                                                seq_len=self.seq_len,
                                                use_schlagnorm=self.use_schlagnorm,
                                                schlagnorm_targets=self.schlagnorm_targets)

        # Two-layer MLP
        if self.use_mlp:
            self.linear = [nn.Dense(self.dim_feedforward, use_bias=self.use_bias,kernel_init=self.initializer),
                           nn.gelu,
                           nn.Dense(self.embed_dim, use_bias=self.use_bias,kernel_init=self.initializer)]
        
        # LayerNorms
        if self.use_layer_norm:
            self.norm1 = nn.LayerNorm()
            self.norm2 = nn.LayerNorm()

    def __call__(self, 
                 x: jnp.ndarray, 
                 interpol_call: bool=False) -> jnp.ndarray:
        '''Applies the Transformer Block to the input tensor.'''
        b,s,_ = x.shape
        mask = jnp.ones_like(x)
        
        if self.mask_inputs:
            mask = mask.at[:, :, :2*self.data_dim].set(jnp.zeros(shape=(b,s,2*self.data_dim)))
            
        x = ((self.mask_inputs) * mask + (not self.mask_inputs)) * x
        
        # LayerNorm Attention
        sa_x = self.norm1(x) if self.use_layer_norm else x

        # Attention mechanism
        attn_out, att, heads = self.self_attn(sa_x, interpol_call)
        x = x + attn_out
        
        # MLP mechanism
        if self.use_mlp:
            # LayerNorm MLP
            mlp_out = self.norm2(x)  if self.use_layer_norm else x
            for l in self.linear:
                mlp_out = l(mlp_out)
            x = x + mlp_out
        
        return (x, (None if self.use_mesa else att), (mlp_out if self.use_mlp else attn_out), heads)

class TF_BlockList(nn.Module):
    '''
    List of Transformer Blocks
    Fields:
        'use_layernorm' (bool): Flag whether to use LayerNorm
        'use_bias' (bool): Flag whether to use bias in the attention layer
        'use_mlp' (bool): Flag whether to use the MLP
        'masked' (bool): Flag whether to use masking in the attention layer
        'interpol' (bool): Flag whether to use interpolation in the attention layer
        'mask_inputs' (bool): Flag whether to mask the input data
        'use_mesa' (bool): Flag whether to use MESA
        'standard_backprop_mesa' (bool): Flag whether to use standard backpropagation in MESA
        'normalization_mesa' (bool): Flag whether to use normalization in MESA
        'num_layers' (int): Number of Transformer Blocks
        'num_heads' (int): Number of attention heads
        'data_dim' (int): Dimension of the input data
        'seq_len' (int): Length of the input sequence
        'embed_dim' (int): Dimension of the embeddings
        'key_size' (int): Size of the key vectors
        'dim_feedforward_MLP' (int): Dimension of the feedforward network
        'use_clip' (bool): Flag whether to clip the output
        'clip' (float): Value to clip the output
        'linear' (bool): Flag whether to use a linear layer
        'initializer' (Callable[[random.PRNGKey, tuple, jnp.dtype], jnp.ndarray]): Initializer for the weights
        'linearize' (bool): Flag whether to linearize the output
        'linear_idx' (int): Index of the linear layer
        'use_schlag_norm' (bool): Flag whether to use Schlag-Norm (used in nonlinear experiments)
        'schlagnorm_targets' (bool): Flag whether to use Schlag-Norm on the targets (used in nonlinear experiments)
    '''
    use_layernorm: bool
    use_bias: bool
    use_mlp: bool
    masked: bool
    interpol: bool
    mask_inputs: bool
    use_mesa: bool
    standard_backprop_mesa: bool
    normalization_mesa: bool
    num_layers: int
    num_heads: int
    data_dim: int
    seq_len: int
    embed_dim: int
    key_size: int
    seq_len: int
    dim_feedforward_MLP: int
    use_clip: bool
    clip: float
    linear: bool
    initializer: Callable[[random.PRNGKey, tuple, jnp.dtype], jnp.ndarray]
    linearize: bool
    linear_idx: int
    use_schlagnorm: bool
    schlagnorm_targets: bool

    def setup(self):
        '''Initializes the list of Transformer Blocks.'''
        self.blocklist = [TransformerBlock(data_dim = self.data_dim,
                                            embed_dim = self.embed_dim,
                                            key_size = self.key_size,
                                            num_heads = self.num_heads,
                                            dim_feedforward = self.dim_feedforward_MLP,
                                            use_layer_norm = self.use_layernorm,
                                            use_bias = self.use_bias,
                                            use_softmax = (not (self.linear or (self.linearize and i == self.linear_idx))),
                                            use_mlp = self.use_mlp,
                                            masked = self.masked,
                                            interpol = self.interpol,
                                            mask_inputs = self.mask_inputs,
                                            use_mesa = self.use_mesa,
                                            standard_backprop_mesa = self.standard_backprop_mesa,
                                            normalization_mesa = self.normalization_mesa,
                                            initializer = self.initializer,
                                            use_pe_kq = False,
                                            seq_len = self.seq_len,
                                            use_schlagnorm = self.use_schlagnorm,
                                            schlagnorm_targets = self.schlagnorm_targets)
                            for i in range(self.num_layers)]

    def __call__(self, 
                 x: jnp.ndarray, 
                 interpol_call: bool=False) -> jnp.ndarray:
        '''
        Applies the list of Transformer Blocks to the input tensor.
        Args:
            x (jnp.ndarray): Input tensor
            interpol_call (bool): Flag whether to use interpolation in the attention layer
        Returns:
            jnp.ndarray: Tuple[List-Result, 
                               Activation after layer i (incl. res), 
                               Attentionmap per layer, 
                               Attention-Result per layer (without residual stream etc.)]
        '''
        output = x
        activations = []
        attention_outputs = []
        attention_maps = []
        heads_list = []
        

        for layer in range(self.num_layers):
            output, att, attn_out, heads = self.blocklist[layer](output, interpol_call=interpol_call)
            if self.use_clip:
                output = jnp.clip(output, (-1.0)*self.clip, self.clip)
            activations.append(output)
            attention_maps.append(att)
            attention_outputs.append(attn_out)
            heads_list.append(heads)
        return output, activations, attention_maps, attention_outputs, heads_list

class FullTransformerModel(nn.Module):
    '''
    Full Transformer Model
    Fields:
        'use_emb' (bool): Flag whether to use an embedding layer
        'use_pe_emb' (bool): Flag whether to use positional encoding in the initial embeddings
        'use_pe_kq' (bool): Flag whether to use (special) positional encoding in the query and key vectors
        'hybrid_first_block' (bool): Flag whether to use a hybrid first block
        'hybrid_sm' (bool): Flag whether to use softmax in the hybrid block
        'num_hybrid_heads' (int): Number of attention heads in the hybrid block
        'input_dim' (int): Dimension of the input data
        'pe_dim' (int): Dimension of the positional encoding
        'out_dim' (int): Dimension of the output data
        'data_dim' (int): Dimension of the input data
        'initializer' (Callable[[random.PRNGKey, tuple, jnp.dtype], jnp.ndarray]): Initializer for the weights
        'use_layernorm' (bool): Flag whether to use LayerNorm
        'use_bias' (bool): Flag whether to use bias in the attention layer
        'use_mlp' (bool): Flag whether to use the MLP
        'masked' (bool): Flag whether to use masking in the attention layer
        'interpol' (bool): Flag whether to use interpolation in the attention layer
        'use_clip' (bool): Flag whether to clip the output
        'mask_inputs' (bool): Flag whether to mask the input data
        'use_mesa' (bool): Flag whether to use MESA
        'standard_backprop_mesa' (bool): Flag whether to use standard backpropagation in MESA
        'normalization_mesa' (bool): Flag whether to use normalization in MESA
        'num_layers' (int): Number of Transformer Blocks
        'num_heads' (int): Number of attention heads
        'embed_dim' (int): Dimension of the embeddings
        'key_size' (int): Size of the key vectors
        'seq_len' (int): Length of the input sequence
        'dim_feedforward_MLP' (int): Dimension of the feedforward network
        'clip' (float): Value to clip the output
        'linear' (bool): Flag whether to use a linear layer
        'linearize' (bool): Flag whether to linearize the output
        'linear_idx' (int): Index of the linear layer
        'use_schlag_norm' (bool): Flag whether to use Schlag-Norm (used in nonlinear experiments) ('..hyb' for hybrid block)
        'schlagnorm_targets' (bool): Flag whether to use Schlag-Norm on the targets (used in nonlinear experiments) ('..hyb' for hybrid block)
    '''
    use_emb: bool = False
    use_pe_emb: bool = False
    use_pe_kq: bool = False
    hybrid_first_block: bool = False
    hybrid_sm: bool = True
    num_hybrid_heads: int = 4
    input_dim: int = 10
    pe_dim: int = 40
    out_dim: int = 10
    data_dim: int = 10
    initializer: Callable[[random.PRNGKey, tuple, jnp.dtype], jnp.ndarray] = jax_net.initializers.normal(stddev = 0.02)
    use_schlagnorm_hyb: bool = False
    schlagnorm_targets_hyb: bool = False
    hybrid_mesa: bool = False

    use_layernorm: bool = False
    use_bias: bool = False
    use_mlp: bool = False
    masked: bool = True
    interpol: bool = False
    use_clip: bool = False
    mask_inputs: bool = False
    use_mesa: bool = False
    standard_backprop_mesa: bool = False
    normalization_mesa: bool = False
    num_layers: int = 1
    num_heads: int = 4
    embed_dim: int = 40
    key_size: int = 10
    seq_len: int = 50
    dim_feedforward_MLP: int = 80
    clip: float = 10.0
    linear: bool = False
    linearize: bool = False
    linear_idx: int = 0
    use_schlagnorm: bool = False
    schlagnorm_targets: bool = False

    def setup(self):
        '''Initializes the Full Transformer Model.'''
        if self.use_pe_emb:
            self.pe = PositionalEncoding(pe_dim = self.pe_dim, max_len=self.seq_len, concat=False)
        if self.hybrid_first_block:
            self.hybrid_block = TransformerBlock(data_dim = self.data_dim,
                                                embed_dim = self.embed_dim,
                                                key_size = self.key_size,
                                                num_heads = self.num_hybrid_heads,
                                                dim_feedforward = self.dim_feedforward_MLP,
                                                use_layer_norm = self.use_layernorm,
                                                use_bias = self.use_bias,
                                                use_softmax = False if self.linearize and self.linear_idx==0 else self.hybrid_sm,
                                                use_mlp = self.use_mlp,
                                                masked = self.masked,
                                                interpol = self.interpol,
                                                mask_inputs = self.mask_inputs,
                                                use_mesa = self.hybrid_mesa,
                                                standard_backprop_mesa = self.standard_backprop_mesa,
                                                normalization_mesa = self.normalization_mesa,
                                                initializer = self.initializer,
                                                use_pe_kq = self.use_pe_kq,
                                                seq_len = self.seq_len,
                                                use_schlagnorm = self.use_schlagnorm_hyb,
                                                schlagnorm_targets = self.schlagnorm_targets_hyb)

        self.tf_block = TF_BlockList(use_layernorm = self.use_layernorm,
                                     use_bias = self.use_bias,
                                     use_mlp = self.use_mlp,
                                     masked = self.masked,
                                     interpol = self.interpol,
                                     use_clip = self.use_clip,
                                     mask_inputs = self.mask_inputs,
                                     use_mesa = self.use_mesa,
                                     standard_backprop_mesa = self.standard_backprop_mesa,
                                     normalization_mesa = self.normalization_mesa,
                                     num_layers = self.num_layers,
                                     num_heads = self.num_heads,
                                     embed_dim = self.embed_dim,
                                     key_size = self.key_size,
                                     seq_len = self.seq_len,
                                     dim_feedforward_MLP = self.dim_feedforward_MLP,
                                     clip = self.clip,
                                     linear = self.linear,
                                     data_dim = self.data_dim,
                                     initializer = self.initializer,
                                     linearize = True if self.linearize and not (self.hybrid_first_block and self.linear_idx == 0) else False,
                                     linear_idx = self.linear_idx - 1 if self.linearize and self.hybrid_first_block else self.linear_idx,
                                     use_schlagnorm = self.use_schlagnorm,
                                     schlagnorm_targets = self.schlagnorm_targets)
        
        if self.use_emb:
            self.input_layer = nn.Dense((self.embed_dim), use_bias=self.use_bias, kernel_init=self.initializer)
            self.output_layer = nn.Dense(self.out_dim, use_bias=self.use_bias, kernel_init=self.initializer)

    def __call__(self, 
                 x: jnp.ndarray, 
                 interpol_call: bool=False) -> jnp.ndarray:
        '''
        Applies the Full Transformer Model to the input tensor.
        Args:
            x (jnp.ndarray): Input tensor
            interpol_call (bool): Flag whether to use interpolation in the attention layer
        Returns:
            jnp.ndarray: Output tensor
        '''

        # Input Embedding:
        if self.use_emb:
            x = self.input_layer(x)
        # Positional Encoding concatenated to Embeddings:
        if self.use_pe_emb:
            x = self.pe(x)
        
        attention_maps = []
        attention_outputs = []
        activations = [x]
        heads_list = []

        # Hybrid First Block (If Full-Softmax, this acts like another normal softmax block)
        if self.hybrid_first_block:
            x, att, attn_out, heads = self.hybrid_block(x, interpol_call)
            if self.use_clip:
                x = jnp.clip(x, (-1.0)*self.clip, self.clip)
            attention_maps.append(att)
            attention_outputs.append(attn_out)
            activations.append(x)
            heads_list.append(heads)
        
        # Transformer Blocks:
        output, activations_block, attention_maps_block, attention_outputs_block, heads_list_block = self.tf_block(x, interpol_call)
        
        activations += activations_block
        attention_maps += attention_maps_block
        attention_outputs += attention_outputs_block
        heads_list += heads_list_block

        if self.use_emb:
            output = self.output_layer(output)

        # Return outputs and tokens after Copy Layer (for copy analysis):
        return output, (activations, attention_maps, attention_outputs, heads_list)
