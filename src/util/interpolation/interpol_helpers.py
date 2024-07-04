
'''
    Module for the interpolation helper functions, including the 
    one-layer reverse algorithm, the weight generator and the TFSubdiagonals class 
    that extracts the mean values of sub-diagonals of the attention matrix-products in a Transformer model.
'''

import jax
import flax
import jax.numpy as jnp
from jax import vmap
from functools import partial
from typing import Tuple, List

from src.data.seq.sequence_data import DataGenerator
from src.train import _compute_loss as compute_loss

class OneLayerRevAlg_Linear:
    '''Class implementing the one-layer reverse algorithm for a linear model.'''
    def __init__(self, 
                 data_dim: int, 
                 seq_len: int):
        '''
        Initializes the one-layer reverse algorithm for a linear model.
        Args:
            'data_dim' (int): Dimension of the data
            'seq_len' (int): Length of the sequence
        '''
        self.data_dim = data_dim
        self.seq_len = seq_len
        self.obs_dim = data_dim

    def compute_terms(self, 
                      x: jnp.ndarray, 
                      x_shifted: jnp.ndarray) -> Tuple[jnp.ndarray]:
        '''Computes the individual terms for the one-layer reverse algorithm'''
        prcr = jnp.matmul(x_shifted[:, :, None], x[:, None, :])
        crpr = jnp.matmul(x[:, :, None], x_shifted[:, None, :])
        prpr = jnp.matmul(x_shifted[:, :, None], x_shifted[:, None, :])
        crcr = jnp.matmul(x[:, :, None], x[:, None, :])
        res = tuple([jnp.cumsum(e, axis=0) for e in (crcr,crpr,prcr,prpr)])
        return res

    @partial(jax.jit, static_argnums=(0))
    def w_func(self, 
               alpha: float, 
               eta: float, 
               p1: float, 
               p2: float, 
               x: float, 
               x_shifted: float) -> jnp.ndarray:
        '''Helper function for the one-layer reverse algorithm'''
        (curr_curr, curr_prev, prev_curr, prev_prev) = self.compute_terms(x, x_shifted)
        return alpha*(p1*curr_curr+p2*curr_prev)+eta*(p1*prev_curr+p2*prev_prev)
    
    def batched_matvec(self, 
                       b_vector: jnp.ndarray, 
                       b_matrix: jnp.ndarray) -> jnp.ndarray:
        '''Helper function for matrix-vector multiplication'''
        return vmap(jnp.matmul, in_axes=(0, 0))(b_vector, b_matrix)

    def formula_model(self, 
                      x: jnp.ndarray, 
                      alpha: float, 
                      eta: float, 
                      beta: float, 
                      gamma: float, 
                      p1: float, 
                      p2: float) -> jnp.ndarray:
        '''Helper function for the one-layer reverse algorithm'''
        x_shifted = jnp.concatenate((jnp.expand_dims(x[0], axis=0)*0, x), axis=0)[:-1]
        return self.batched_matvec(x, self.w_func(alpha, eta, p1, p2, x, x_shifted)) + self.batched_matvec(x_shifted, self.w_func(beta, gamma, p1, p2, x, x_shifted))
    
    def batched_formula_model(self):
        '''Helper function for the one-layer reverse algorithm'''
        return vmap(self.formula_model, in_axes=(0, None, None, None, None, None, None))

    def revAlg_one_layer(self, 
                         batch: jnp.ndarray, 
                         alpha1: float, 
                         eta1: float, 
                         beta1: float, 
                         gamma1: float, 
                         alpha2: float, 
                         eta2: float, 
                         beta2: float,
                         gamma2: float, 
                         p1: float, 
                         p2: float, 
                         p3: float, 
                         p4: float) -> jnp.ndarray:
        '''Computes the one-layer reverse algorithm for a one layer linear Transformer model.'''
        return self.batched_formula_model()(batch, alpha1, eta1, beta1, gamma1, p1, p2) + self.batched_formula_model()(batch, alpha2, eta2, beta2, gamma2, p3, p4)

    def evaluate_revAlg(self, 
                        revAlgargs:Tuple[float],
                        test_rng:jax.random.PRNGKey,
                        data_generator:DataGenerator) -> float:
        '''Evaluates the one-layer reverse algorithm for a one layer linear Transformer model.'''
        total_loss = 0
        for _ in range(10):
            test_rng, batch_rng = jax.random.split(test_rng)
            batch_test,_ = data_generator.get_data(rng=batch_rng, batch_size=1000)
            revAlg_pred = self.revAlg_one_layer(batch_test[0][:,:,self.obs_dim:self.obs_dim*2], *revAlgargs)
            total_loss += compute_loss(revAlg_pred, batch_test[1])
        return total_loss / 10


class WeightGenerator:
    '''Class implementing the weight generator for a Transformer model.'''
    def __init__(self, data_dim: int):
        '''
        Initializes the weight generator for a Transformer model.
        Args:
            'data_dim' (int): Dimension of the data
        '''
        self.data_dim = data_dim
        
    def two_head_ks20(self, a:float, e:float, b:float, g:float, a2:float, e2:float, b2:float, g2:float, p1:float, p2:float, p3:float, p4:float) -> Tuple[jnp.ndarray]:
        '''Constructs the Q, K, V, and P matrices for a multi-head attention mechanism for given subdiagonal-parameters.'''
        d = self.data_dim
        Q1 = jnp.block([[jnp.zeros((2*d, 2*d))],
                        [a*jnp.eye(d),e*jnp.eye(d)],
                        [b*jnp.eye(d),g*jnp.eye(d)]])
        Q2 = jnp.block([[jnp.zeros((2*d, 2*d))],
                        [a2*jnp.eye(d),e2*jnp.eye(d)],
                        [b2*jnp.eye(d),g2*jnp.eye(d)]])
        # K:
        K1, K2 = tuple([jnp.block([[jnp.zeros((2*d, 2*d))],
                        [jnp.eye(d), jnp.zeros((d,d))],
                        [jnp.zeros((d,d)), jnp.eye(d)]]) for _ in range(2)])
        # K:
        V1 = jnp.block([[jnp.zeros((2*d, 2*d))],
                        [p1*jnp.eye(d),jnp.zeros((d,d))],
                        [p2*jnp.eye(d),jnp.zeros((d,d))]])
        V2 = jnp.block([[jnp.zeros((2*d, 2*d))],
                        [p3*jnp.eye(d),jnp.zeros((d,d))],
                        [p4*jnp.eye(d),jnp.zeros((d,d))]])
        # Projection:
        P1, P2 = tuple([jnp.block([[jnp.eye(2*d),jnp.zeros((2*d, 2*d))]]) for _ in range(2)])
        Q = jnp.concatenate((Q1, Q2), axis=1)
        K = jnp.concatenate((K1, K2), axis=1)
        V = jnp.concatenate((V1, V2), axis=1)
        P = jnp.concatenate((P1, P2), axis=0)
        return Q,K,V,P

    def create_block_matrix(self, diagonal_values:List[float], d:int) -> jnp.ndarray:
        '''
        Creates a block matrix from a list of diagonal values. 
        Each value is used to create a dxd diagonal matrix, and these are arranged in a 4x4 block structure.
        Args:
            'diagonal_values' (List[float]): List of diagonal values
            'd' (int): Dimension of the sub-diagonal matrices
        Returns:
            (jnp.ndarray): Constructed Kernel
        '''
        return jnp.block([[diagonal_values[i*4 + j]*jnp.eye(d) for j in range(4)] for i in range(4)])

    def four_head_ks40(self, d1:List[float], d2:List[float], d3:List[float], d4:List[float], p1:List[float], p2:List[float], p3:List[float], p4:List[float], data_dim:int) -> Tuple[jnp.ndarray]:
        '''
        Constructs the Q, K, V, and P matrices for a 4-head attention mechanism with key_size 40.
        Each input vector is used to create a corresponding block matrix.
        '''
        d = data_dim
        Q = jnp.concatenate(tuple([self.create_block_matrix(d_vals, d) for d_vals in [d1, d2, d3, d4]]), axis=1)
        K = jnp.concatenate(tuple([jnp.eye(4*d) for _ in range(4)]), axis=1)
        V = jnp.concatenate(tuple([self.create_block_matrix(p_vals, d) for p_vals in [p1, p2, p3, p4]]), axis=1)
        P = jnp.concatenate(tuple([jnp.eye(4*d) for _ in range(4)]), axis=0)
        
        return Q, K, V, P

class TFSubdiagonals:
    
    def two_head_diags(params:flax.core.frozen_dict.FrozenDict, key_size:int) -> Tuple[Tuple[jnp.ndarray]]:
        '''Special function for input masked construction experiments with one layer Transformers'''
        q_weights, k_weights, v_weights, p_weights = tuple([params['tf_block']['blocklist_0']['self_attn'][name]['kernel'] \
                                                            for name in ['q_proj','k_proj','v_proj','o_proj']])
        
        W_q1, W_q2 = jnp.split(q_weights, 2, axis=1)
        W_k1, W_k2 = jnp.split(k_weights, 2, axis=1)
        W_v1, W_v2 = jnp.split(v_weights, 2, axis=1)
        P = p_weights

        KtQ1 = jnp.matmul(W_q1, W_k1.T)
        PWv1 = jnp.matmul(W_v1, P[0:key_size,:])
        KtQ2 = jnp.matmul(W_q2,W_k2.T)
        PWv2 = jnp.matmul(W_v2, P[key_size:2*key_size,:])


        # TODO: This is a super ugly solution, but it works for now, but will need to be refactored
        d1 = jnp.mean(jnp.diagonal(KtQ1[20:30,20:30]))
        d2 = jnp.mean(jnp.diagonal(KtQ1[20:30,30:40]))
        d3 = jnp.mean(jnp.diagonal(KtQ1[30:40,20:30]))
        d4 = jnp.mean(jnp.diagonal(KtQ1[30:40,30:40]))

        d5 = jnp.mean(jnp.diagonal(KtQ2[20:30,20:30]))
        d6 = jnp.mean(jnp.diagonal(KtQ2[20:30,30:40]))
        d7 = jnp.mean(jnp.diagonal(KtQ2[30:40,20:30]))
        d8 = jnp.mean(jnp.diagonal(KtQ2[30:40,30:40]))

        p1 = jnp.mean(jnp.diagonal(PWv1[20:30,0:10]))
        p2 = jnp.mean(jnp.diagonal(PWv1[30:40,0:10]))
        p3 = jnp.mean(jnp.diagonal(PWv2[20:30,0:10]))
        p4 = jnp.mean(jnp.diagonal(PWv2[30:40,0:10]))

        return ((d1,d2,d3,d4,d5,d6,d7,d8),(p1,p2,p3,p4))

    def four_head_diags_multilayer(params: flax.core.frozen_dict.FrozenDict, 
                                   num_layers: int, 
                                   key_size: int) -> List[Tuple[Tuple[jnp.ndarray]]]:
        '''Extracts the mean values of the sub-diagonals of the attention matrix-products in a multi-layer Transformer model.'''
        def four_head_diags(params: flax.core.frozen_dict.FrozenDict, 
                        layer: int, 
                        key_size: int) -> Tuple[Tuple[jnp.ndarray]]:
        
            def calculate_mean_diagonal_submatrices(KtQ: jnp.ndarray) -> List[float]:
                '''Calculates the mean values of the sub-diagonals of the attention matrix-products in a Transformer model.'''
                d_list = []
                for i in range(4):  
                    for j in range(4):  
                        submatrix = KtQ[i*10:(i+1)*10, j*10:(j+1)*10]  
                        d = jnp.mean(jnp.diagonal(submatrix)) 
                        d_list.append(d)
                return d_list
            
            access_name = f'blocklist_{layer}'
            W_q, W_k, W_v, p_weights = tuple([params['tf_block'][access_name]['self_attn'][name]['kernel'] \
                                                                for name in ['q_proj','k_proj','v_proj','o_proj']])
            
            P = p_weights

            KtQs = []
            PWvs = []

            W_q_sub = list(jnp.split(W_q, 4, axis=1))
            W_k_sub = list(jnp.split(W_k, 4, axis=1))
            W_v_sub = list(jnp.split(W_v, 4, axis=1))

            for i in range(4):
                KtQs.append(jnp.matmul(W_q_sub[i], W_k_sub[i].T))
                PWvs.append(jnp.matmul(W_v_sub[i], P[i*key_size:(i+1)*key_size, :]))

            d1, d2, d3, d4 = tuple([tuple(calculate_mean_diagonal_submatrices(KtQ)) for KtQ in KtQs])
            p1, p2, p3, p4 = tuple([tuple(calculate_mean_diagonal_submatrices(PWv)) for PWv in PWvs])

            return ((d1, d2, d3, d4),(p1,p2,p3,p4))
    
        result = []
        for layer in range(num_layers):
            (d,p) = four_head_diags(params=params, layer=layer, key_size=key_size)
            result.append((d,p))
        return result
