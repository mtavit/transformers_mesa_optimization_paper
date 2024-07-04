
'''Module for auxiliary models.'''
import jax
import jax.numpy as jnp
import abc
from functools import partial
from flax.training import train_state
from tqdm import tqdm
from typing import Callable, List, Tuple
from ml_collections import config_dict

from src.train import _compute_loss
from src.optim import Optimizer
from src.data.seq.sequence_data import DataGenerator

#################################################
#                                               #
#                                               #
#       Implementation of Matrix-Inv.           #
#                Approximators                  #
#                                               #
#                                               #
#################################################

def invert_matrix_neumann(A: jnp.ndarray,
                          steps: int, 
                          norm: float) -> jnp.ndarray:
    '''
    Function to approximate a matrix inverse using a truncated Neumann series.
    Args:
        'A' (jnp.ndarray): The matrix to be inverted.
        'steps' (int): The number of steps for the Neumann series.
        'norm' (float): The norm-scalar for enabling a neumann matrix approximation.
    Returns:
        jnp.ndarray: The approximation of the inverted matrix.
    '''
    n = A.shape[0]
    A = A / norm
    I = jnp.eye(n)
    diff = I - A
    inverse_approx = I
    term = I
    for _ in range(steps):
        term = term @ diff
        inverse_approx += term
    return inverse_approx/norm

def batched_neumann(steps: int, 
                    norm: float) -> Callable[[jnp.ndarray], jnp.ndarray]:
    '''Function to batch the Neumann series approximation of a matrix inverse.'''
    return jax.vmap(partial(invert_matrix_neumann, 
                            steps=steps, 
                            norm=norm), 
                    in_axes=(0))

def invert_matrix_newton(A: jnp.ndarray, 
                         steps: int) -> jnp.ndarray:
    '''
    Function to approximate a matrix inverse using Newton's method.
    Args:
        'A' (jnp.ndarray): The matrix to be inverted.
        'steps' (int): The number of steps for Newton's method.
    Returns:
        jnp.ndarray: The approximation of the inverted matrix.
    '''
    n = A.shape[0]
    X = jnp.eye(n) / jnp.trace(A)
    for _ in range(steps):
        AX = jnp.dot(A, X)
        X = X @ (2 * jnp.eye(n) - AX)
    return X

def batched_newton(steps: int) -> Callable[[jnp.ndarray], jnp.ndarray]:
    '''Function to batch the Newton's method approximation of a matrix inverse.'''
    return jax.vmap(partial(invert_matrix_newton, 
                            steps=steps), 
                    in_axes=(0))

def invert_matrix_chebyshev(A: jnp.ndarray,
                            steps: int,
                            alphas: jnp.ndarray,
                            betas: jnp.ndarray) -> jnp.ndarray:
    norm = jnp.linalg.norm(A)
    n = A.shape[0]
    A = A/norm
    I = jnp.eye(n)
    diff = I - A
    inverse_approx = I
    term = I
    prev = I
    for alpha, beta in zip(alphas, betas):
        diff = I - alpha*A
        term = term@diff
        term_momentum = beta*(inverse_approx - prev)
        prev = inverse_approx
        inverse_approx = inverse_approx + term + term_momentum
    return inverse_approx/norm

def batched_chebyshev(steps: int,
                      alphas: List[float], 
                      betas: List[float]) -> Callable[[jnp.ndarray], jnp.ndarray]:
    return jax.vmap(partial(invert_matrix_chebyshev,
                            steps=steps,
                            alphas=alphas,
                            betas=betas),
                    in_axes=(0))

def invert_matrix_richardson(A, omegas):
    norm = jnp.linalg.norm(A)
    A = A/norm
    I = jnp.eye(A.shape[0])
    diff = I - A
    inverse_approx = I
    term = I

    for omega in omegas:
        diff = I - omega*A
        term = term@diff
        inverse_approx = inverse_approx + term
    return inverse_approx/norm

def batched_richardson(omegas):
    return jax.vmap(partial(invert_matrix_richardson, 
                            omegas=omegas),
                    in_axes=0)

#################################################
#                                               #
#                                               #
#             Learn parameters for              #
#            chebyshev-inverse-apx.             #
#                                               #
#                                               #
#################################################

def learn_parameters_chebyshev(num_steps: int, 
                               train_len: int, 
                               experiment_config: config_dict.ConfigDict,
                               data_generator: DataGenerator,
                               part_obs_constr: bool = False,
                               part_obs_embed_dim: int = 80,
                               seq_len: int = 50,
                               use_mlp: bool = False,
                               init_alphas: jnp.ndarray = None,
                               init_betas: jnp.ndarray = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    '''Training logic for GD to learn optimal parameters for Chebyshev, evaluated using Sequencesolver on autoregressive tasks of choice.'''
    def single_cheb_pred(params_s, seq_data_s, seq_labels_s, lamb_s):
            preds = []
            for token in range(seq_len):
                inv_mat = seq_data_s[:token].T@seq_data_s[:token] + lamb_s*jnp.eye(seq_data_s[:token].shape[1])
                w_hat = invert_matrix_chebyshev(A=inv_mat,
                                            steps=num_steps, 
                                            alphas=params_s['params']['alphas'], 
                                             betas=params_s['params']['betas']) @ (seq_data_s[:token].T@seq_labels_s[:token])
                if use_mlp:
                    test_token = data_generator._mini_mlp(seq_labels_s[token])
                    test_token = test_token/(jnp.linalg.norm(test_token)+1e-16)
                else:
                    test_token = seq_labels_s[token]
                seq_label_hat = jnp.matmul(test_token, w_hat)
                preds.append(seq_label_hat)
            return jnp.array(preds)

    def cheb_lsq_pred(params, seq_data, seq_labels, lamb):
        vectorized_cheb_pred = jax.vmap(single_cheb_pred, in_axes=(None, 0, 0, None))(params, seq_data, seq_labels, lamb)
        return vectorized_cheb_pred
    
    def get_features(batch: jnp.ndarray) -> jnp.ndarray:
        feature_seq_func = lambda seq : jax.vmap(data_generator._mini_mlp)(seq)
        feature_batch = jax.vmap(feature_seq_func, in_axes=(0))(batch)
        return feature_batch
    
    def cheb_loss(params, rng):
        rng, batch_rng = jax.random.split(rng)
        batch, _ = data_generator.get_data(rng=batch_rng, batch_size=experiment_config.data.batch_size)
        data, labels = batch
        if part_obs_constr:
            batch_size, seq_len, obs_dim = data.shape
            constructed_data = jnp.zeros(shape=(batch_size, seq_len, part_obs_embed_dim))
            constructed_data = constructed_data.at[:,:,0:obs_dim].set(data)
            for k in range(1, part_obs_embed_dim//obs_dim):
                shifted_data = jnp.concatenate((jnp.zeros(shape=(batch_size,(k),obs_dim)),data[:,:-1*(k),:]),axis=1)
                constructed_data = constructed_data.at[:,:,k*obs_dim:(k+1)*obs_dim].set(shifted_data)
            shifted_data = jnp.concatenate([jnp.expand_dims(constructed_data[:, 0, :], 1)*0, constructed_data], axis=1)[:, :-1, :]
            data = constructed_data
            preds_chebyshev = cheb_lsq_pred(params, shifted_data, data, 0.001)[:,:,0:obs_dim]
        elif use_mlp:
            dat_feat = get_features(data)
            dat_feat /= (jnp.linalg.norm(dat_feat,axis=-1)[...,None])
            shifted_data = jnp.concatenate([jnp.expand_dims(dat_feat[:, 0, :], 1)*0, dat_feat], axis=1)[:, :-1, :]
            preds_chebyshev = cheb_lsq_pred(params, shifted_data, data/(jnp.linalg.norm(data, axis=-1)[...,None]), 0.001)
        else:
            shifted_data = jnp.concatenate([jnp.expand_dims(data[:, 0, :], 1)*0, data], axis=1)[:, :-1, :]
            preds_chebyshev = cheb_lsq_pred(params, shifted_data, data, 0.001)
        loss = _compute_loss(preds=preds_chebyshev, targets=labels)
        return loss, rng

    def cheb_train_step(state, rng):
        loss_fn = lambda params: cheb_loss(params=params, rng=rng)
        (loss, rng), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, (loss, rng)
    fast_cheb_train_step = jax.jit(cheb_train_step)

    def cheb_training(state):
        for i in range(train_len):
            rng=jax.random.PRNGKey(seed=i)
            state, (loss,_) = fast_cheb_train_step(state, rng)
            print(loss)
            if i % 1000 == 0:
                print(state.params)
        return state

    init_params = {'params': {}}
    init_params['params']['alphas'] = jnp.ones(shape=(num_steps,)) if init_alphas == None else init_alphas
    init_params['params']['betas'] = jnp.zeros(shape=(num_steps,)) if init_betas == None else init_betas

    optimizer = Optimizer().get_optimizer()

    state_cheb = train_state.TrainState.create(apply_fn=cheb_lsq_pred, params=init_params, tx=optimizer)
    state_cheb = cheb_training(state_cheb)

    return state_cheb.params['params']['alphas'], state_cheb.params['params']['betas'] 


#################################################
#                                               #
#                                               #
#       Implementation of Aux-Models            #
#                                               #
#                                               #
#################################################

class AuxModel(metaclass=abc.ABCMeta):
    '''Abstract Base Class for auxiliary models'''

    @abc.abstractmethod
    def predict(self, shifted_data, data):
        '''Abstract method to get prediction for data batch'''
        raise NotImplementedError

class LeastSquaresSequenceSolver(AuxModel):
    '''Class implementing a least squares solver for sequence data.'''
    def __init__(self,  
                 approximator: str,
                 seq_len: int,
                 apx_steps: int = 6,
                 apx_norm: float = 70,
                 lamb: float = 0.001,
                 use_mlp: bool = False,
                 mlp_fn = None,
                 alphas = None,
                 betas = None):
        '''
        Initializes the LeastSquaresSequenceSolver.
        Args:
            'approximator' (str): The approximator to use for inverting the matrix.
            'seq_len' (int): The sequence length.
            'apx_steps' (int): The number of steps for the approximator.
            'apx_norm' (float): The norm scalar for the matrix approximation.
            'lamb' (float): The lambda parameter for the least squares solver.
        '''
        if not approximator == None:
            if approximator not in ['neumann', 'newton', 'chebyshev', 'richardson', 'None']:
                raise ValueError(f"Approximator {approximator} not supported")
        
        if approximator == 'neumann':
            self.inverter = lambda A : invert_matrix_neumann(A, apx_steps, apx_norm)
        elif approximator == 'newton':
            self.inverter = lambda A : invert_matrix_newton(A, apx_steps)
        elif approximator == 'richardson':
            self.inverter = lambda A : invert_matrix_richardson(A, omegas=alphas)
        elif approximator == 'chebyshev':
            self.inverter = lambda A : invert_matrix_chebyshev(A, apx_steps, alphas=alphas, betas=betas)
        else:
            self.inverter = jnp.linalg.inv 

        self.seq_len = seq_len
        self.apx_steps = apx_steps
        self.apx_norm = apx_norm
        self.lamb = lamb
        self.use_mlp = use_mlp
        self.mlp_fn = mlp_fn

    def predict(self, 
                shifted_data: jnp.ndarray, 
                data: jnp.ndarray) -> jnp.ndarray:
        '''
        Function to get predictions for a batch of data.
        Args:
            'shifted_data' (jnp.ndarray): The shifted data.
            'data' (jnp.ndarray): The original data.
        Returns:
            jnp.ndarray: The predictions.
        '''
        return self.all_preds(seq_data=shifted_data,
                              seq_labels=data,
                              seq_len=self.seq_len,
                              lamb=self.lamb)

    def least_squares_one_iter(self, 
                               seq_data: jnp.ndarray, 
                               seq_labels: jnp.ndarray, 
                               lamb: float) -> jnp.ndarray:
        '''Function to perform one iteration of the least squares solver.'''
        return self.inverter(seq_data.T@seq_data + lamb*jnp.eye(seq_data.shape[1])) @ (seq_data.T@seq_labels)

    def least_squares_seq_pred_single_seq(self, 
                               seq_data: jnp.ndarray, 
                               seq_labels: jnp.ndarray, 
                               seq_len: int, 
                               lamb: float) -> jnp.ndarray:
        '''
        Function to get predictions for a single sequence using the least squares solver.
        Args:
            'seq_data' (jnp.ndarray): The sequence data.
            'seq_labels' (jnp.ndarray): The sequence labels.
            'seq_len' (int): The sequence length.
            'lamb' (float): The lambda parameter for the least squares solver.
        Returns:
            jnp.ndarray: The predictions.
        '''
        preds = []
        for token in range(seq_len):
            w_hat = self.least_squares_one_iter(seq_data[:token], seq_labels[:token], lamb=lamb)
            if self.use_mlp:
                test_token = self.mlp_fn(seq_labels[token])
                test_token = test_token/(jnp.linalg.norm(test_token)+1e-16)
            else:
                test_token =seq_labels[token]
            seq_label_hat = jnp.matmul(test_token, w_hat)
            preds.append(seq_label_hat)
        return jnp.array(preds)

    def all_preds(self, 
                  seq_data: jnp.ndarray, 
                  seq_labels: jnp.ndarray, 
                  seq_len: int,
                  lamb: float) -> jnp.ndarray:
        '''Function to get predictions for all sequences using the least squares solver.'''
        return jax.vmap(self.least_squares_seq_pred_single_seq, in_axes=(0,0,None,None))(seq_data, seq_labels, seq_len, lamb)
    
    def get_features(self, batch: jnp.ndarray) -> jnp.ndarray:
        feature_seq_func = lambda seq : jax.vmap(self.mlp_fn)(seq)
        feature_batch = jax.vmap(feature_seq_func, in_axes=(0))(batch)
        return feature_batch

    def opt_lamb(self, 
                 minv: float, 
                 maxv: float, 
                 steps: int, 
                 data_generator: DataGenerator,
                 loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                 part_obs_constr: bool,
                 embed_dim: int = 80,
                 constr: bool = False,
                 slots: bool = 4) -> jnp.ndarray:
        '''
        Function to optimize lambda parameter for least squares solver via line-search.
        Sets the lambda parameter to the value that minimizes the loss function and returns it
        Args:
            'minv' (float): The minimum value for the lambda parameter.
            'maxv' (float): The maximum value for the lambda parameter.
            'steps' (int): The number of steps for the line-search.
            'data_generator' (DataGenerator): The data generator.
            'loss_fn' (Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]): The loss function. 
            'part_obs_constr' (bool): Use part.-obs. construction (concat. past k tokens)
            'embed_dim' (int): Construction size for partobs
            'constr' (bool): Use Full-Obs. construction,
            'slots' (int): Token-'slots' in Full-Obs. construction
        Returns:
            jnp.ndarray: The optimized lambda parameter.
        '''
        min_score = float('inf')
        range_vals = jnp.linspace(minv, maxv, steps)
        rng = jax.random.PRNGKey(42)
        rng, test_rng = jax.random.split(rng)

        for lam in range_vals:
            (data, targets), _ = data_generator.get_data(rng=test_rng, batch_size=512)

            if part_obs_constr:
                batch_size, seq_len, obs_dim = data.shape
                constructed_data = jnp.zeros(shape=(batch_size, seq_len, embed_dim))
                constructed_data = constructed_data.at[:,:,0:obs_dim].set(data)
                for k in range(1, embed_dim//obs_dim):
                    shifted_data = jnp.concatenate((jnp.zeros(shape=(batch_size,(k),obs_dim)),data[:,:-1*(k),:]),axis=1)
                    constructed_data = constructed_data.at[:,:,k*obs_dim:(k+1)*obs_dim].set(shifted_data)
                shifted_data = jnp.concatenate([jnp.expand_dims(constructed_data[:, 0, :], 1)*0, constructed_data], axis=1)[:, :-1, :]
                data = constructed_data
                preds_lsq = self.all_preds(seq_data=shifted_data, 
                                           seq_labels=data, 
                                           seq_len=self.seq_len, 
                                           lamb=lam)[:,:,0:obs_dim]
            elif self.use_mlp:
                dat_feat = self.get_features(data)
                dat_feat /= (jnp.linalg.norm(dat_feat,axis=-1)[...,None])
                shifted_data = jnp.concatenate([jnp.expand_dims(dat_feat[:, 0, :], 1)*0, dat_feat], axis=1)[:, :-1, :]
                preds_lsq = self.all_preds(seq_data=shifted_data, 
                                           seq_labels=data/(jnp.linalg.norm(data, axis=-1)[...,None] + 1e-16), 
                                           seq_len=self.seq_len, 
                                           lamb=lam)
            else:
                if constr:
                    shifted_data = data[:,:,(slots-1)*targets.shape[-1]:]
                    data= data[:,:,(slots-2)*targets.shape[-1]:(slots-1)*targets.shape[-1]]
                else:
                    shifted_data = jnp.concatenate([jnp.expand_dims(data[:, 0, :], 1)*0, data], axis=1)[:, :-1, :]
    
                preds_lsq = self.all_preds(seq_data=shifted_data, 
                                           seq_labels=data, 
                                           seq_len=self.seq_len, 
                                           lamb=lam)

            score = loss_fn(preds_lsq, targets)
            print(f"for lambda = {lam:.6f} lsq-loss: ", score)
            if score < min_score:
                min_score = score
                self.lamb = lam
        return self.lamb

class GDSequenceSolver:
    '''Class implementing a gradient descent solver for sequence data.'''
    def __init__(self, 
                 eta: float, 
                 lamb: float = 0,
                 seq_len: int = 50):
        '''
        Initializes the GDSequenceSolver.
        Args:
            'eta' (float): The learning rate.
            'lamb' (float): The (optional) lambda parameter.
        '''
        self.eta = eta
        self.lamb = lamb
        self.seq_len = seq_len

    def predict(self, 
                shifted_data: jnp.ndarray, 
                data: jnp.ndarray) -> jnp.ndarray:
        '''
        Function to get predictions for a batch of data.
        Args:
            'shifted_data' (jnp.ndarray): The shifted data.
            'data' (jnp.ndarray): The original data.
        Returns:
            jnp.ndarray: The predictions.
        '''
        return self.all_preds(seq=data,
                              seq_shifted=shifted_data,
                              eta=self.eta)

    def gd_delta(self, 
           seq: jnp.ndarray, 
           seq_shifted: jnp.ndarray,
           idx: int) -> jnp.ndarray:
        '''Function to get the delta for the gradient descent solver.'''
        outer_productsGD = jnp.matmul(seq_shifted[:, :, None], seq[:, None, :])
        resultGD = jnp.cumsum(outer_productsGD, axis=0)
        return resultGD[idx]

    def one_step_gd(self, 
                    seq: jnp.ndarray, 
                    seq_shifted: jnp.ndarray, 
                    eta: float, 
                    lamb: float,
                    seq_len: int) -> jnp.ndarray:
        '''
        Function to perform one step of the gradient descent solver.
        Args:
            'seq' (jnp.ndarray): The sequence data.
            'seq_shifted' (jnp.ndarray): The shifted sequence data.
            'eta' (float): The learning rate.
            'lamb' (float): The lambda parameter.
            'seq_len' (int): The sequence length of the test sequences.
        Returns:
            jnp.ndarray: The gradient descent updates.
        '''
        result = []
        for idx in range(seq_len):
            deltaWi = self.gd_delta(seq=seq, 
                                    seq_shifted=seq_shifted, 
                                    idx=idx)
            deltaWi += lamb*deltaWi
            gd_update = eta * (seq[idx] @ deltaWi)
            result.append(gd_update)
        return jnp.array(result)

    def all_preds(self, seq: jnp.ndarray, seq_shifted: jnp.ndarray, eta: float) -> jnp.ndarray:
        '''Function to get predictions for all sequences in a batch using the gradient descent solver.'''
        return jax.vmap(self.one_step_gd, in_axes=(0,0,None,None,None))(seq, seq_shifted, eta, self.lamb, self.seq_len)
    
    def opt_eta(self, 
                minv: float, 
                maxv: float, 
                steps: int, 
                data_generator: DataGenerator,
                loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                constr: bool = False,
                slots: bool = 4) -> jnp.ndarray:
        '''
        Function to optimize eta parameter for GD sequence solver via line-search.
        Sets the eta parameter to the value that minimizes the loss function and returns it
        Args:
            'minv' (float): The minimum value for the lambda parameter.
            'maxv' (float): The maximum value for the lambda parameter.
            'steps' (int): The number of steps for the line-search.
            'data_generator' (DataGenerator): The data generator.
            'loss_fn' (Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]): The loss function. 
            'constr' (bool): Use Full-Obs. construction,
            'slots' (int): Token-'slots' in Full-Obs. construction
        Returns:
            jnp.ndarray: The optimized lambda parameter.
        '''
        min_score = float('inf')
        range_vals = jnp.linspace(minv, maxv, steps)
        rng = jax.random.PRNGKey(42)
        rng, test_rng = jax.random.split(rng)

        for eta in range_vals:
            (data, targets), _ = data_generator.get_data(rng=test_rng, batch_size=512)
            if constr:
                shifted_data = data[:,:,(slots-1)*targets.shape[-1]:]
                data= data[:,:,(slots-2)*targets.shape[-1]:(slots-1)*targets.shape[-1]]
            else:
                shifted_data = jnp.concatenate([jnp.expand_dims(data[:, 0, :], 1)*0, data], axis=1)[:, :-1, :]

            preds_gd = self.all_preds(seq=data, 
                                      seq_shifted=shifted_data, 
                                      eta=eta)
            
            score = loss_fn(preds_gd, targets)
            print(f"for eta = {eta:.6f} lsq-loss: ", score)
            if score < min_score:
                min_score = score
                self.eta = eta
        return self.eta