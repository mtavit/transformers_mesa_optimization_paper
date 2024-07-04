import math
import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial
from typing import Optional
from src.models.positional_encoding import PositionalEncoding

#TODO: @Johannes Please update with your improved Mesa-Code 

def scaled_dot_product_attention(q, k, v, use_softmax=False, normalization=False, masked=True):
    # [BS, SEQ, X_Y_DIM]
    d_k = q.shape[-1]                                            # get dimension of q,k,v
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))         # compute inner product
    attn_logits = attn_logits                                    # divide by dim. to keep var
    if masked:
        # Autoregressive mask
        mask = jnp.tril(jnp.ones(shape=(attn_logits.shape)), k=0)     # Creates Lower Tri-Mat. with 0s everywhere else
        attn_logits = jnp.where(mask == 0, 0, attn_logits)   # add mask (for Linear)
        if use_softmax:
            attn_logits = jnp.where(mask == 0, -1e30, attn_logits/math.sqrt(d_k))   # add mask (for SoftMax)
    if use_softmax:
        attn_logits = nn.softmax(attn_logits, axis=-1)
    elif normalization:
        attn_norms = jnp.linalg.norm(attn_logits, axis=-1)
        attn_logits = attn_logits / (1e-16 + attn_norms[..., None])
    values = jnp.matmul(attn_logits, v)                           # matrix prod with values
    return values

def explicit_shermo(cur_inverse, x_T, forward=1):
    """Shermon-Morrison update on rank one matrix x^T*x
        Use forward=-1 to compute the reverse direction.
    """
    # compute X^TH since we need this two times
    x_TH = x_T[..., None].T@cur_inverse
    #scalar in Sherman–Morrison formula
    Hx = cur_inverse@x_T[..., None]

    scalar = forward + x_T[..., None].T@Hx
    #Inverse update based on Sherman–Morrison formula
    return cur_inverse - Hx@x_TH/scalar

def new_q_prediction(cur_inverse, x_T, q):
    """Multiply current S matrix with q i.e. (XX^T-lambda I)^{-1}q"""
    new_inverse = explicit_shermo(cur_inverse, x_T)
    # prediction based on new W
    return new_inverse @ q, new_inverse

def run_shermo_pred_forward(carry, token):
    """Iteratively compute S matrix and multiply with q's"""
    cur_inverse = carry
    (x_T, q) = token
    shermo_pred, new_inverse = new_q_prediction(cur_inverse, x_T, q)
    return (new_inverse), (shermo_pred)

@partial(jax.vmap,in_axes=(0,0,0, None, None))
def mesa_layer(x, y, q, lambda_s, normalization=False):
    """Iteratively compute S matrix and multiply with q's"""
    init_inverse = jnp.eye(x.shape[-1])*lambda_s
    stacked = (x, q)
    _, preds = jax.lax.scan(run_shermo_pred_forward, (init_inverse), stacked)
    return scaled_dot_product_attention(preds, x, y,
                                        normalization=normalization)

def _compute_new_q_shermo(x, q, lambda_s):
    init_inverse = jnp.eye(x.shape[-1])*lambda_s
    stacked = (x, q)
    (H), preds = jax.lax.scan(run_shermo_pred_forward, (init_inverse), stacked)
    return preds, (H, x, q)

@jax.custom_vjp
def compute_custom_mesa_layer(x, q, lambda_s):
    init_inverse = jnp.eye(x.shape[-1])*lambda_s
    stacked = (x, q)
    return  jax.lax.scan(run_shermo_pred_forward, (init_inverse), stacked)[1]

def compute_mesa_custom_fwd(x, q, lambda_s):
    return _compute_new_q_shermo(x, q, lambda_s)

def compute_new_A(cur_A, prev_inverse, g_t, x_t, q_t):
    return jax.grad(lambda H: (cur_A*explicit_shermo(H, x_t)).sum()+  (g_t*new_q_prediction(H, x_t, q_t)[0]).sum())(prev_inverse)

def compute_mesa_custom_bwd(res, g):
    (H, x, q) = res
    def accumulate_jacobian(carry, token):
        """This function computes iteratively the backproped error."""
        cur_inverse, cur_A = carry
        x_t, q_t, g_t = token

        # compute the previous inverse and data correlation matrix (backward mode)
        prev_inverse = explicit_shermo(cur_inverse, x_t, forward=-1)

        # the direct gradients at time t
        direct_grad_q, direct_grad_x = jax.grad(lambda q_, x_: (g_t*new_q_prediction(prev_inverse, x_, q_)[0]).sum(), argnums=(0,1))(q_t, x_t)

        # the indirect gradient wrt to x_t making use of the backward computed A
        indirect_grad_x = jax.grad(lambda x: (cur_A*explicit_shermo(prev_inverse, x)).sum())(x_t)

        new_grad_x = direct_grad_x + indirect_grad_x
        new_grad_q = direct_grad_q

        # compute the feedbacked quantities
        new_A = compute_new_A(cur_A, prev_inverse, g_t, x_t, q_t)

        return (prev_inverse, new_A), (new_grad_x, new_grad_q)

    # concate tokens so we can pass them through easily in the scan method
    x_ = jnp.flip(x, 0)
    q_ = jnp.flip(q, 0)
    g_ = jnp.flip(g, 0)
    (original_inverse, new_A), (new_grad_x, new_grad_q) = jax.lax.scan(accumulate_jacobian, (H, jnp.zeros_like(H)), (x_, q_, g_))

    # # flip to correct order of how the it was presented to the sequence
    new_grad_x = jnp.flip(new_grad_x, 0)
    new_grad_q = jnp.flip(new_grad_q, 0)
    return (new_grad_x,
            new_grad_q,
            jnp.trace(new_A))

compute_custom_mesa_layer.defvjp(compute_mesa_custom_fwd, compute_mesa_custom_bwd)

@partial(jax.vmap,in_axes=(0,0,0, None, None))
def mesa_layer_custom_bwd(x, y, q, lambda_s, normalization=False):
    preds = compute_custom_mesa_layer(x, q, lambda_s)
    return scaled_dot_product_attention(preds, x, y, normalization=normalization)


class MultiHeadMesa(nn.Module):
    """Multi-headed Mesa (MHME) module.
    This module is intended to apply the Mesa layer over sequences of vectors.
    Rough sketch:
    - Compute queries (Q), inputs (X), and targets (Y) as projections of inputs.
    - For every query q_t at element t which we intepret as time,
        compute W_t = argmin ||WX[:t] - Y[:t]||^2 + \lambda||W||^2. Note that this
        respects causal masking logic i.e. does not depend on elements of the
        sequence > t.
    - Output PQ with Q = [W_0q_0, \dots, W_Tq_T] and P a projection matrix.
    Glossary of shapes:
    - T: Sequence length.
    - D: Vector (embedding) size.
    - H: Number of attention heads.
    """

    num_heads: int
    input_size: int
    emb_size: int
    seq_len: int
    initializer: jax.nn.initializers
    use_bias_p: bool = False
    shift_with_x0: bool = True
    standard_backprop: bool = True
    normalization: bool = False
    name: str = None
    use_schlagnorm: bool = False
    schlagnorm_targets: bool = False
    use_pe_kq: bool = False

    def setup(self):
        """Initialises the module.
        Args:
            num_heads: Number of independent mesa heads (H).
            input_size: The size of inputs, targets and queries (suboptimal naming - this is the same as "key size").
            w_init: Initialiser for weights in the linear maps.
            emb_size: The size of the embeddings (also called model size).
            seq_len: Length of input sequences
            use_bias_p: Use bias parameters in the linear operations of the network.
            key_shift: Shift the keys of the key matrix by one.
            shift_with_x0: When shifting, it is not clear what to do with the first token. We could shift the token itself or by 0.
            standard_backprop: Use standard autodiff backprop or custom version.
            normalization: Normalize (to norm 1) K^Tq product
            name: Optional name for this module.
        """

        self.q_proj = nn.Dense(self.input_size*self.num_heads,
                                        kernel_init=self.initializer,
                                        use_bias=self.use_bias_p
                                    )
        self.v_proj = nn.Dense(self.input_size*self.num_heads,
                                    kernel_init=self.initializer,
                                    use_bias=self.use_bias_p
                                )
        self.k_proj = nn.Dense(self.input_size*self.num_heads,
                                    kernel_init=self.initializer,
                                    use_bias=self.use_bias_p
                                )
        self.o_proj = nn.Dense(self.emb_size,
                                kernel_init=self.initializer,
                                use_bias=self.use_bias_p
                                )

        if self.standard_backprop:
            self.mesa =  mesa_layer
        else:
            self.mesa = mesa_layer_custom_bwd
        
        if self.use_pe_kq:
            self.pos_enc_kq = PositionalEncoding(pe_dim = self.emb_size, max_len=self.seq_len, concat=True)

    @nn.compact
    def __call__(
        self,
        x,
        mask: Optional[jnp.ndarray] = None,
        ) -> jnp.ndarray:
        """Computes MHSM with queries, inputs & targets.
        This module broadcasts over zero or more 'batch-like' leading dimensions.
        Args:
            query: Embeddings sequence used to compute queries; shape [..., T', D_q].
            input: Embeddings sequence used to compute inputs; shape [..., T, D_k].
            target: Embeddings sequence used to compute targets; shape [..., T, D_v].
            mask: Has no effect. Only here for compatibility.
        Returns:
            A new sequence of embeddings, consisting of a projection of the
            mesa layer updated projections; shape [..., T', D'].
        """
        query = x
        input = x
        target = x
        *leading_dims, _ = query.shape

        # Compute query/input/target per head
        if self.use_pe_kq:
            t = self.pos_enc_kq(x)
            target_heads = self.k_proj(t).reshape((*leading_dims, self.num_heads, self.input_size))
            query_heads = self.q_proj(t).reshape((*leading_dims, self.num_heads, self.input_size))
        else:
            query_heads = self.q_proj(query).reshape((*leading_dims, self.num_heads, self.input_size))
            target_heads = self.k_proj(target).reshape((*leading_dims, self.num_heads, self.input_size))
        input_heads = self.v_proj(input).reshape((*leading_dims, self.num_heads, self.input_size))
        
        qkv = query_heads.transpose((0, 2, 1, 3)), target_heads.transpose((0, 2, 1, 3)), input_heads.transpose((0, 2, 1, 3))

        # Optional: Normalize using SchlagNorm
        if self.use_schlagnorm:
            query_heads = query_heads/(1e-16+jnp.linalg.norm(query_heads, axis=-1)[...,None])
            target_heads = target_heads/(1e-16+jnp.linalg.norm(target_heads, axis=-1)[...,None])
            if self.schlagnorm_targets:
                target_heads = target_heads/(1e-16+jnp.linalg.norm(input_heads, axis=-1)[...,None])

        lambdas = self.param('lambdas', jax.nn.initializers.constant(1.), (self.num_heads,))
        mesa_out = jax.vmap(self.mesa, in_axes=(2,2,2,0, None), out_axes=-1 )(input_heads,
                                                                        target_heads,
                                                                        query_heads,
                                                                        lambdas,
                                                                        self.normalization)
        # Apply a final projection to project back into the embedding size
        return self.o_proj(mesa_out.reshape(*leading_dims,self.num_heads* self.input_size )), None, qkv