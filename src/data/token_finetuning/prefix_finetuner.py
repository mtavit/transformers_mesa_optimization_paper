
'''
    Fine-tuner class for fine-tuning the a prefix-prompt
    for in-context regression data prediction tasks.
'''

import optax
import jax.numpy as jnp
from jax import random, jit, value_and_grad
from flax import linen as nn
from flax.training import train_state
from functools import partial
from typing import Tuple, Dict

from src.data.token_finetuning.abs_finetuner import FineTuner
from src.data.datagenerator import DataGenerator

class PrefixFineTuner(FineTuner):
    '''Fine-tuner class for fine-tuning the prefix-prompt for in-context regression data prediction tasks.'''
    def __init__(self, 
                 model_tf: nn.Module, 
                 state_tf: train_state.TrainState, 
                 data_generator: DataGenerator,
                 train_batch_size: int,
                 finetune_steps: int,
                 eos_token: jnp.ndarray,
                 prefix_dims: Tuple[int]):
        '''
        Initializes the Prefix Fine-Tuner.
        Args:
            'model_tf' (nn.Module): The Flax (Transformer) Model.
            'state_tf' (train_state.TrainState): The Flax TrainState for the model.
            'data_generator' (DataGenerator): The DataGenerator.
            'train_batch_size' (int): The batch size for finetuning.
            'finetune_steps' (int): The number of finetuning steps.
            'eos_token' (jnp.ndarray): The end-of-sentence token (Has to be finetuned previously).
            'prefix_dims' (Tuple[int]): The dimensions of the prefix-prompt.
        '''
        super().__init__(model_tf=model_tf, 
                         state_tf=state_tf, 
                         data_generator=data_generator,
                         train_batch_size=train_batch_size,
                         finetune_steps=finetune_steps)
        self.eos_token = eos_token
        self.prefix_dims = prefix_dims
    
    def finetune(self, 
                 rng: random.PRNGKey, 
                 finetune_lr: float = 9e-4) -> jnp.ndarray:
        '''
        Fine-tunes the prefix-prompt (P) for in-context regression data prediction tasks.
        Args:
            'rng' (random.PRNGKey): The random number generator key.
            'finetune_lr' (float): The learning rate for finetuning.
        Returns:
            jnp.ndarray: The finetuned prefix-prompt.
        '''
        rng_init_prompt, rng_finetune = random.split(rng, 2)
        prompt = random.normal(rng_init_prompt, shape=self.prefix_dims)
        optimizer_prompt = optax.adam(finetune_lr)
        state_finetune = train_state.TrainState.create(apply_fn=self.get_tf_pred,
                                                       params={'prefix': {'prompt' : prompt}},
                                                       tx=optimizer_prompt)
        tf_params = self.state_tf.params
        for epoch in range(self.finetune_steps):
            state_finetune, loss, rng_finetune = self.train_step_ft(rng=rng_finetune,
                                                                    state_finetune=state_finetune,
                                                                    tf_params=tf_params)
            
            if epoch % 100 == 0:
                print('epoch ', epoch, ':', loss)
        
        return state_finetune.params['prefix']['prompt']
    

    def get_tf_pred(self, 
                    rng: random.PRNGKey, 
                    tf_params: Dict[str, any], 
                    prompt: jnp.ndarray) -> jnp.ndarray:
        '''
        Returns the predictions of the model for a given prompt.
        Args:
            'rng' (random.PRNGKey): The random number generator key.
            'tf_params' (Dict[str, any]): The parameters of the model.
            'prompt' (jnp.ndarray): The prefix-prompt.
        Returns:
            Tuple[jnp.ndarray]: The predictions of the model and the actuals target labels.
        '''
        _,_,(batch,labels) = self.data_generator.get_data(rng=rng, 
                                                          batch_size=self.train_batch_size, 
                                                          eos=self.eos_token, 
                                                          prompt=prompt, 
                                                          use_prompt=True)
        logits, _ = self.model_tf.apply({'params': tf_params}, batch, interpol_call=False)
        return logits, labels

    def loss_fn(self,
                rng: random.PRNGKey, 
                tf_params: Dict[str, any],
                finetune_params: Dict[str, any]) -> jnp.ndarray:
        '''
        Returns the loss for the fine-tuning.
        Args:
            'rng' (random.PRNGKey): The random number generator key.
            'tf_params' (Dict[str, any]): The parameters of the model.
            'finetune_params' (Dict[str, any]): The parameters for fine-tuning.
        Returns:
            jnp.ndarray: The loss for the fine-tuning the prefix prompt.
        '''
        preds, labels = self.get_tf_pred(rng, tf_params, finetune_params['prefix']['prompt'])
        bs,sl,_ = preds.shape
        ps = finetune_params['prefix']['prompt'].shape[0]
        return jnp.sum((labels[:,ps:,:][:,0::3,:] - preds[:,ps:,:][:,0::3,:])**2)/(2*bs*(sl/3))

    @partial(jit, static_argnums=(0))
    def train_step_ft(self, 
                      rng: random.PRNGKey, 
                      state_finetune: train_state.TrainState, 
                      tf_params: Dict[str, any]) -> Tuple[train_state.TrainState, jnp.ndarray, random.PRNGKey]:
        '''
        Executes a finetuning step.
        Args:
            'rng' (random.PRNGKey): The random number generator key.
            'state_finetune' (train_state.TrainState): The TrainState for the finetuning.
            'state_tf' (train_state.TrainState): The TrainState for the model.
        Returns:
            Tuple[train_state.TrainState, jnp.ndarray, random.PRNGKey]: 
                    - The updated TrainState for the finetuning.
                    - The loss for the finetuning.
                    - The random number generator key.
        '''
        rng, next_rng = random.split(rng)
        loss, grad = value_and_grad(self.loss_fn, argnums=2)(rng, tf_params, state_finetune.params)
        state_finetune = state_finetune.apply_gradients(grads=grad)
        return state_finetune, loss, next_rng