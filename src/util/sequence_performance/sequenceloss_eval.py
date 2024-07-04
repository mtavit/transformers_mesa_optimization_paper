
'''Module for evaluating the performance of sequence prediction models.'''

import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from jax import vmap, random
from typing import List,Callable, Dict
from tqdm import tqdm

import abc

from src.models.auxiliary_models import AuxModel
from src.data.datagenerator import DataGenerator

class EvalModel(abc.ABC):
    '''Abstract base class for evaluation models.'''
    @abc.abstractmethod
    def evaluate(self, 
                 data: jnp.ndarray, 
                 targets: jnp.ndarray, 
                 loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
        pass

class TFEvaluator(EvalModel):
    '''Class for evaluating transformer models.'''
    def __init__(self, 
                 model: nn.Module, 
                 state: train_state.TrainState,
                 constr: bool,
                 slots: int):
        '''
        Initializes the TFEvaluator.
        Args:
            'model' (nn.Module): The model to evaluate.
            'state' (train_state.TrainState): The state of the model.
            'constr' (bool): Whether the model is using constructed data.
            'slots' (int): The number of slots.
        '''
        self.model = model
        self.state = state
        self.constr = constr
        self.slots = slots

    def evaluate(self, 
                 data:jnp.ndarray, 
                 targets:jnp.ndarray, 
                 loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
        '''
        Evaluates the model.
        Args:
            'data' (jnp.ndarray): The input data.
            'targets' (jnp.ndarray): The target data.
            'loss_fn' (Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]): The loss function.
        Returns:
            'jnp.ndarray': The loss.
        '''
        logits, _ = self.model.apply({'params': self.state.params}, data, interpol_call=False)
        preds = logits[:, :, :targets.shape[-1]]
        return loss_fn(preds, targets)

class AuxmodelEvaluator(EvalModel):
    '''Class for evaluating auxiliary models.'''
    def __init__(self,
                 model: AuxModel,
                 state: any,
                 constr: bool,
                 slots: int,
                 part_obs: bool,
                 use_mlp: bool):	
            '''
            Initializes the AuxmodelEvaluator.
            Args:
                'model' (AuxModel): The model to evaluate.
                'state' (int): Used for Part.-Obs. Construction (holds embed_dim) or mlp (holds mlp_function)
                'constr' (bool): Whether the model is using constructed data.
                'slots' (int): The number of slots.
                'part_obs' (bool): Evaluate on constructed token of past partial observations
                'use_mlp' (bool): Evaluate on mlp features

            '''
            self.model = model
            self.state = state
            self.constr = constr
            self.slots = slots
            self.part_obs = part_obs
            self.use_mlp = use_mlp
    
    def get_features(self, batch: jnp.ndarray) -> jnp.ndarray:
        feature_seq_func = lambda seq : vmap(self.state)(seq)
        feature_batch = vmap(feature_seq_func, in_axes=(0))(batch)
        return feature_batch
    
    def evaluate(self, 
                 data:jnp.ndarray, 
                 targets:jnp.ndarray, 
                 loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
        '''
        Evaluates the model.
        Args:
            'data' (jnp.ndarray): The input data.
            'targets' (jnp.ndarray): The target data.
            'loss_fn' (Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]): The loss function.
        Returns:
            'jnp.ndarray': The loss.
        '''
        if self.part_obs:
            batch_size, seq_len, obs_dim = data.shape
            embed_dim = self.state
            constructed_data = jnp.zeros(shape=(batch_size, seq_len, embed_dim))
            constructed_data = constructed_data.at[:,:,0:obs_dim].set(data)
            for k in range(1, embed_dim // obs_dim):
                shifted_data = jnp.concatenate((jnp.zeros(shape=(batch_size,(k),obs_dim)),data[:,:-1*(k),:]),axis=1)
                constructed_data = constructed_data.at[:,:,k*obs_dim:(k+1)*obs_dim].set(shifted_data)
            shifted_data = jnp.concatenate([jnp.expand_dims(constructed_data[:, 0, :], 1)*0, constructed_data], axis=1)[:, :-1, :]
            data = constructed_data
            preds = self.model.predict(shifted_data=shifted_data, data=data)[:,:,0:obs_dim]
        elif self.use_mlp:
            dat_feat = self.get_features(data)
            dat_feat /= (jnp.linalg.norm(dat_feat,axis=-1)[...,None])
            shifted_data = jnp.concatenate([jnp.expand_dims(dat_feat[:, 0, :], 1)*0, dat_feat], axis=1)[:, :-1, :]
            preds = self.model.predict(shifted_data=shifted_data, data=data/(jnp.linalg.norm(data, axis=-1)[...,None]))
        else:
            if self.constr:
                shifted_data = data[:,:,(self.slots-1)*targets.shape[-1]:]
                data = data[:,:,(self.slots-2)*targets.shape[-1]:(self.slots-1)*targets.shape[-1]]
            else:
                shifted_data = jnp.concatenate([jnp.expand_dims(data[:, 0, :], 1)*0, data], axis=1)[:, :-1, :]
            preds = self.model.predict(shifted_data=shifted_data, data=data)
        return loss_fn(preds, targets)

def get_evaluator(model_type:str, **kwargs) -> EvalModel:
    '''Returns an evaluator based on the model type.'''
    if model_type.lower() in ['transformer', 'mesa-transformer']:
        return TFEvaluator(**kwargs)
    elif model_type.lower() in ['lsq', 'gd', 'lsq_partobs', 'lsq_mlp']:
        return AuxmodelEvaluator(part_obs=(model_type == 'lsq_partobs'), use_mlp=(model_type == 'lsq_mlp'), **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

class SequencePredictionEvaluator:
    '''Class for evaluating sequence prediction models across test sequences, per token.'''
    def __init__(self, 
                 data_generator: DataGenerator,
                 test_batch_size: int,
                 seeds: List[int],
                 model_list: List[str],
                 models: List[nn.Module],
                 states: List[train_state.TrainState],
                 loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]):
        '''
        Initializes the SequencePredictionEvaluator.
        Args:
            'data_generator' (DataGenerator): The data generator.
            'test_batch_size' (int): The batch size for testing.
            'seeds' (List[int]): The seeds for testing.
            'model_list' (List[str]): The list of model types.
            'models' (List[nn.Module]): The models to evaluate.
            'states' (List[any]): The states of the models.
            'loss_fn' (Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]): The loss function for evaluation.
        '''
        self.data_generator = data_generator
        self.test_batch_size = test_batch_size
        self.seeds = seeds
        self.model_list = model_list
        self.models = models
        self.states = states
        self.loss_fn = loss_fn
        self.constr = data_generator.get_data_info()['constr']
        if self.constr:
            self.slots = data_generator.get_data_info()['slots']
        else:
            self.slots = 0
        self.losses = [[] for _ in range(len(self.model_list))]

    def run(self) -> Dict[str, any]:
        '''Runs the evaluation experiment.'''
        for seed in self.seeds:
            test_rng = random.PRNGKey(seed)
            (data, targets), _ = self.data_generator.get_data(rng=test_rng, batch_size=self.test_batch_size)
            for modelname, idx in zip(self.model_list, jnp.arange(len(self.model_list))):
                evaluator = get_evaluator(model_type=modelname, model=self.models[idx], state=self.states[idx], constr=self.constr, slots=self.slots)
                result = evaluator.evaluate(data, targets, self.loss_fn)
                self.losses[idx].append(result)

        return {'losses': self.losses}