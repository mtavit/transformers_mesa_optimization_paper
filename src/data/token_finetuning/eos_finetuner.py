
'''
    Fine-tuner class for fine-tuning the an end-of-sentence token (EOS)
    for in-context regression data prediction tasks.
'''

import optax
import jax.numpy as jnp
from jax import random, jit, value_and_grad
from flax import linen as nn
from flax.training import train_state
from functools import partial
from typing import Dict, Tuple

from src.data.token_finetuning.abs_finetuner import FineTuner
from src.data.datagenerator import DataGenerator

class EOSFineTuner(FineTuner):
    '''Fine-tuner class for fine-tuning the an end-of-sentence token (EOS) for in-context regression data prediction tasks.'''
    def __init__(self, 
                 model_tf: nn.Module, 
                 state_tf: train_state.TrainState, 
                 data_generator: DataGenerator,
                 train_batch_size: int,
                 finetune_steps: int,
                 eos_dim: int,
                 use_prompt: bool,
                 prompt: jnp.ndarray = None):
        '''
        Initializes the EOS Fine-Tuner.
        Args:
            'model_tf' (nn.Module): The Flax (Transformer) Model.
            'state_tf' (train_state.TrainState): The Flax TrainState for the model.
            'data_generator' (DataGenerator): The DataGenerator.
            'train_batch_size' (int): The batch size for finetuning.
            'finetune_steps' (int): The number of finetuning steps.
            'eos_dim' (int): The dimension of the end-of-sentence token.
        '''
        
        super().__init__(model_tf=model_tf, 
                         state_tf=state_tf, 
                         data_generator=data_generator,
                         train_batch_size=train_batch_size,
                         finetune_steps=finetune_steps)
        self.eos_dim = eos_dim
        self.use_prompt = use_prompt
        self.prompt = prompt
    
    def finetune(self, 
                 rng: random.PRNGKey, 
                 finetune_lr: float = 1e-3) -> jnp.ndarray:
        '''
        Fine-tunes the end-of-sentence token (EOS) for in-context regression data prediction tasks.
        Args:
            'rng' (random.PRNGKey): The random number generator key.
            'finetune_lr' (float): The learning rate for finetuning.
        Returns:
            jnp.ndarray: The finetuned end-of-sentence token.
        '''
        rng_init_eos, rng_finetune = random.split(rng, 2)
        eos = random.normal(rng_init_eos,shape=(self.eos_dim,))
        optimizer_token = optax.adam(finetune_lr)
        state_finetune = train_state.TrainState.create(apply_fn=self.get_tf_pred,
                                                       params={'tokens': {'eos' : eos}},
                                                       tx=optimizer_token)
        tf_params = self.state_tf.params
        for epoch in range(self.finetune_steps):
            state_finetune, loss, rng_finetune = self.train_step_ft(rng=rng_finetune,
                                                                    state_finetune=state_finetune,
                                                                    tf_params=tf_params)
            
            if epoch % 100 == 0:
                print('epoch ', epoch, ':', loss)
        
        return state_finetune.params['tokens']['eos']
    

    def get_tf_pred(self, 
                    rng: random.PRNGKey, 
                    tf_params: Dict[str, any], 
                    eos) -> jnp.ndarray:
        '''
        Executes the Transformer model with the current finetuned data.
        Args:
            'rng' (random.PRNGKey): The random number generator key.
            'tf_params' (Dict[str, any]): The Transformer model parameters.
            'eos' (jnp.ndarray): The end-of-sentence token.
        Returns:
            Tuple[jnp.ndarray]: The predictions of the Transformer model and the actual target labels.
        '''
        (batch, labels), _, _ = self.data_generator.get_data(rng=rng, 
                                                                batch_size=self.train_batch_size, 
                                                                eos=eos, 
                                                                use_prompt=self.use_prompt,
                                                                prompt=self.prompt)
        logits, _ = self.model_tf.apply({'params': tf_params}, batch, interpol_call=False)
        return logits, labels

    def loss_fn(self, 
                rng: random.PRNGKey, 
                tf_params: Dict[str, any],
                finetune_params: Dict[str, any]) -> jnp.ndarray:
        '''
        Calculates the finetuning loss.
        Args:
            'rng' (random.PRNGKey): The random number generator key.
            'tf_params' (Dict[str, any]): The Transformer model parameters.
            'finetune_params' (Dict[str, any]): The finetuned parameters.
        Returns:
            jnp.ndarray: The finetuning loss.
        '''
        preds, labels = self.get_tf_pred(rng, tf_params, finetune_params['tokens']['eos'])
        bs,sl,_ = preds.shape
        return jnp.sum((labels[:,0::3,:] - preds[:,0::3,:])**2)/(2*bs*(sl/3))

    @partial(jit, static_argnums=(0))
    def train_step_ft(self, 
                      rng: random.PRNGKey, 
                      state_finetune: train_state.TrainState, 
                      tf_params: Dict[str, any]) -> Tuple[train_state.TrainState, jnp.ndarray, random.PRNGKey]:
        '''
        Executes a finetuning step.
        Args:
            'rng' (random.PRNGKey): The random number generator key.
            'state_finetune' (train_state.TrainState): The finetuned TrainState.
            'state_tf' (train_state.TrainState): The TrainState for the Transformer model.
        Returns:
            Tuple[train_state.TrainState, jnp.ndarray, random.PRNGKey]: 
                    - The updated finetuned TrainState.
                    - The finetuning loss.
                    - The updated random number generator key.

        '''
        rng, next_rng = random.split(rng)
        loss, grad = value_and_grad(self.loss_fn, argnums=2)(rng, tf_params, state_finetune.params)
        state_finetune = state_finetune.apply_gradients(grads=grad)
        return state_finetune, loss, next_rng