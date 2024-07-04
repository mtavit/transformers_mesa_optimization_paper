
'''Module for evaluating the model on the ICL dataset.'''

import jax.numpy as jnp
from jax import vmap, random
from flax import linen as nn
from flax.training import train_state
from ml_collections import config_dict

from src.data.icl.icl_data import ICLDataGenerator
from src.models.auxiliary_models import AuxModel


class ICLEvaluator:
    '''Class for evaluating the model on the ICL dataset'''
    def __init__(self,
                 experiment_config: config_dict.ConfigDict,
                 model_tf: nn.Module,
                 state_tf: train_state.TrainState,
                 lsq_model: AuxModel,
                 data_generator_single: ICLDataGenerator,
                 data_generator_multi: ICLDataGenerator,
                 prompt: jnp.ndarray,
                 eos: jnp.ndarray):
        '''
        Initializes the ICL evaluator
        Args:
            experiment_config: Configuration for the experiment
            model_tf: Transformer model
            state_tf: Transformer state
            lsq_model: Least squares model
            data_generator_single: Data generator for single task
            data_generator_multi: Data generator for multiple tasks
            prompt: Prompt for the ICL dataset
            eos: End of sequence token for the ICL dataset
        '''
        
        self.experiment_config = experiment_config
        self.model_tf = model_tf
        self.state_tf = state_tf
        self.lsq_model = lsq_model
        self.data_generator_single = data_generator_single
        self.data_generator_multi = data_generator_multi
        self.prompt = prompt
        self.eos = eos
        self.task1_len = data_generator_multi.seq_len_task1


    def run_single_task_icl(self):
        '''Runs the single task ICL evaluation'''
        data_config = self.experiment_config.data

        base_preds, eos_preds, prompt_eos_preds, lsq_preds = [], [], [], []

        for seed in self.experiment_config.seeds:
            test_rng = random.PRNGKey(seed)
            ((data_eos,labels_eos), (data,labels), (data_prompt_eos, labels_prompt_eos)) = self.data_generator_single.get_data(rng=test_rng, 
                                                                                                                                batch_size=data_config.test_batch_size,
                                                                                                                                use_prompt=True,
                                                                                                                                prompt=self.prompt,
                                                                                                                                eos=self.eos)
            data_dim = data.shape[-1]
            bs = data.shape[0]

            shifted_data = jnp.concatenate([jnp.expand_dims(data[:, 0, :], 1)*0, data], axis=1)[:, :-1, :]

            preds_base, preds_eos, preds_prompt_eos = tuple([self.model_tf.apply({'params': self.state_tf.params}, d, interpol_call=False)[0][:,:,:data_dim] 
                                                       for d in [data, data_eos, data_prompt_eos]])

            preds_lsq = self.lsq_model.predict(shifted_data=shifted_data,data=data)

            results = tuple([list((jnp.sum(((pred - target)**2), axis=(0,2))/(2*bs))) for pred,target in zip([preds_base, preds_eos, preds_prompt_eos, preds_lsq],[labels, labels_eos, labels_prompt_eos, labels])])
            prompt=self.prompt

            base_preds.append(results[0][0::2])
            eos_preds.append(results[1][0::3])
            prompt_eos_preds.append(results[2][prompt.shape[0]:][0::3])
            lsq_preds.append(results[3][0::2])

        means = tuple([jnp.mean(jnp.array(pred_list),axis=0)[:self.experiment_config.experiment.experiment_length] for pred_list in [base_preds, eos_preds, prompt_eos_preds, lsq_preds]])
        stds = tuple([jnp.std(jnp.array(pred_list),axis=0)[:self.experiment_config.experiment.experiment_length] for pred_list in [base_preds, eos_preds, prompt_eos_preds, lsq_preds]])

        
        return {'base_preds_mean': means[0],
                'base_preds_std': stds[0],
                'eos_preds_mean': means[1],
                'eos_preds_std': stds[1],
                'prompt_eos_preds_mean': means[2],
                'prompt_eos_preds_std': stds[2],
                'lsq_preds_mean': means[3],
                'lsq_preds_std': stds[3]}


    def run_two_task_icl(self):
        '''Runs the two task ICL evaluation'''
        data_dim = self.data_generator_multi.get_data_info()['data_dim']

        base_multi_preds, eos_multi_preds, prompt_eos_multi_preds, lsq_multi_preds = [], [], [], []
        data_config = self.experiment_config.data
        bs, data_dim = data_config.batch_size, data_config.data_dim

        for seed in self.experiment_config.seeds:
            test_rng = random.PRNGKey(seed)
            t = self.experiment_config.experiment.full_experiment_length
            ((data_eos,labels_eos), (data,labels), (data_prompt_eos, labels_prompt_eos)) = self.data_generator_multi.get_data(rng=test_rng, 
                                                                                                                              batch_size=data_config.test_batch_size,
                                                                                                                              use_prompt=True,
                                                                                                                              prompt=self.prompt,
                                                                                                                              eos=self.eos)
            (d1,t1) = (data_eos[:,:t,:], labels_eos[:,:t,:])
            (d2,t2) = (data[:,:t,:], labels[:,:t,:])
            (d3,t3) = (data_prompt_eos[:,:t,:], labels_prompt_eos[:,:t,:])

            shifted_data = jnp.concatenate([jnp.expand_dims(d2[:, 0, :], 1)*0, d2], axis=1)[:, :-1, :]
            preds_eos, preds_base, preds_prompt_eos = tuple([self.model_tf.apply({'params': self.state_tf.params}, d, interpol_call=False)[0][:,:,:data_dim]
                                                       for d in [d1, d2, d3]])
            
            preds_lsq = self.lsq_model.predict(shifted_data=shifted_data, data=data)
            results = tuple([list((jnp.sum(((pred - target)**2), axis=(0,2))/(2*bs))) for pred,target in zip([preds_base, preds_eos, preds_prompt_eos, preds_lsq],[t2, t1, t3, t2])])
            prompt=self.prompt

            # base_results = list((jnp.sum(((results[0] - t2)**2), axis=(0,2))/(2*bs)))
            # eos_results = list((jnp.sum(((results[1] - t1)**2), axis=(0,2))/(2*bs)))
            # prompt_eos_result = list((jnp.sum(((results[2] - t3)**2), axis=(0,2))/(2*bs)))
            # lsq_result = list((jnp.sum(((results[3] - t2)**2), axis=(0,2))/(2*bs)))

            base_results, eos_results, prompt_eos_result, lsq_result = results

            task1_len = self.task1_len
            prompt_len = prompt.shape[0]

            base_multi_preds.append(base_results[:((task1_len-1-prompt_len)//3)*2][0::2] + base_results[task1_len-1:][0::2])
            eos_multi_preds.append(eos_results[:task1_len-1-prompt_len][0::3] + eos_results[task1_len-1:][0::3])
            prompt_eos_multi_preds.append(prompt_eos_result[:task1_len-1][prompt_len:][0::3] + prompt_eos_result[task1_len-1:][prompt_len:][0::3])
            lsq_multi_preds.append(lsq_result[:((task1_len-1-prompt_len)//3)*2][0::2] + lsq_result[task1_len-1:][0::2])
        
        means = tuple([jnp.mean(jnp.array(pred_list),axis=0) for pred_list in [base_multi_preds, eos_multi_preds, prompt_eos_multi_preds, lsq_multi_preds]])
        stds = tuple([jnp.std(jnp.array(pred_list),axis=0) for pred_list in [base_multi_preds, eos_multi_preds, prompt_eos_multi_preds, lsq_multi_preds]])

        return {'base_preds_mean': means[0],
                'base_preds_std': stds[0],
                'eos_preds_mean': means[1],
                'eos_preds_std': stds[1],
                'prompt_eos_preds_mean': means[2],
                'prompt_eos_preds_std': stds[2],
                'lsq_preds_mean': means[3],
                'lsq_preds_std': stds[3]}