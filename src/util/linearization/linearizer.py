
'''Module for Linearization of Transformer Models'''

import flax.linen as nn
import pickle
from jax import random, nn, jit, numpy as jnp, lax
from flax.training import train_state
from functools import partial
from ml_collections import config_dict
from typing import Dict, List

from src.data.seq.linear_seq import LinearSequenceDataGenerator
from src.data.constr_data import ConstructedFullSeqGenerator
from src.train import Training
from src.attn import MultiHeadAttention
from src.models.full_fledged_transformer import FullTransformerModel
from src.optim import Optimizer
from src.util.linearization.linearization_helpers import LinearizationHelper

class Linearizer:
    '''Class for linearization experiments.'''
    def __init__(self,
                 model_config: config_dict.ConfigDict,
                 experiment_config: config_dict.ConfigDict,
                 constr: bool,
                 special_lrs: List[float],
                 linear_lrs: List[float]):
        '''
        Initializes the Linearizer.
        Args:
            'model_config' (config_dict.ConfigDict): The model configuration.
            'experiment_config' (config_dict.ConfigDict): The experiment configuration.
            'constr' (bool): Whether to use constructed data.
            'special_lrs' (List[float]): The special learning rates.
            'linear_lrs' (List[float]): The linear learning rates.
        '''
        
        self.model_config = model_config
        self.experiment_config = experiment_config

        self.seeds = self.experiment_config.seeds
        
        self.multiseed_losses_tf = []
        self.multiseed_linearization_losses = []
        self.multiseed_linearized_test_losses = []

        self.special_lrs = special_lrs
        self.linear_lrs = linear_lrs

        self.constr = constr

    
    def get_linaerizable_softmaxTF(self, 
                                   input_dim:int, 
                                   embed_dim:int, 
                                   key_size:int, 
                                   seq_len:int, 
                                   data_dim:int, 
                                   linearize:bool, 
                                   linear_idx:int) -> FullTransformerModel:
        '''
            Returns a FullTransformerModel with the specified parameters. 
            If linearize is True, the model will be linearized at the specified linear_idx.
            Args: 
                'input_dim' (int): The input dimension.
                'embed_dim' (int): The embedding dimension.
                'key_size' (int): The key size.
                'seq_len' (int): The sequence length.
                'data_dim' (int): The data dimension.
                'linearize' (bool): Whether to linearize the model.
                'linear_idx' (int): The index of the layer to linearize.
            Returns:
                FullTransformerModel: The FullTransformerModel following the specifications.
        '''
        return FullTransformerModel(use_emb=self.model_config.use_emb,
                                    use_pe_emb=self.model_config.use_pe_emb,
                                    use_pe_kq=self.model_config.use_pe_kq,
                                    hybrid_first_block=self.model_config.hybrid_first_block,
                                    input_dim=input_dim,
                                    out_dim=input_dim,
                                    data_dim=data_dim,
                                    initializer=self.model_config.initializer,
                                    masked=self.model_config.masked,
                                    interpol=self.model_config.interpol,
                                    use_clip=self.model_config.use_clip,
                                    clip=self.model_config.clip_val,
                                    mask_inputs=self.model_config.mask_inputs,
                                    num_layers=self.model_config.num_layers,
                                    num_heads=self.model_config.num_heads,
                                    embed_dim=embed_dim,
                                    key_size=key_size,
                                    seq_len=seq_len,
                                    dim_feedforward_MLP=self.model_config.dim_feedforward_MLP,
                                    linear=False,
                                    linearize=linearize,
                                    linear_idx=linear_idx)
    
    def get_data_generator(self, 
                           data_dim:int, 
                           seq_len:int,
                           embed_dim:int) -> LinearSequenceDataGenerator:
        '''
            Returns a DataGenerator with the specified parameters.
            Args:
                'data_dim' (int): The data dimension.
                'seq_len' (int): The sequence length.
                'emb_dim' (int): Size of the constructed embeddings e.g.: [0,x_i,x_i,x_{i-1}]
            Returns:
                ConstructedFullSeqGenerator: The ConstructedFullSeqGenerator following the specifications.
        '''
        linseq_d =  LinearSequenceDataGenerator(seq_len=seq_len,
                                                data_dim=data_dim,
                                                obs_dim=data_dim,
                                                range=self.experiment_config.data.range,
                                                noise=self.experiment_config.data.noise,
                                                noise_obs=0.0,
                                                data_clip=10000, #self.experiment_config.data.data_clip,
                                                eye_obs=True)
        return ConstructedFullSeqGenerator(data_generator=linseq_d,
                                            embed_dim=embed_dim) if self.constr else linseq_d

    def run(self) -> Dict[str, any]:
        '''Runs the linearization experiments.'''
        num_dims = len(self.experiment_config.experiment.dimensions)
        num_layers = self.model_config.num_layers + (1 if self.model_config.hybrid_first_block else 0)
        num_seeds = len(self.seeds)
        results = jnp.zeros(shape=(num_seeds, num_dims, num_layers))

        for seed_idx, seed in enumerate(self.seeds):
            softmax_lrs, softmax_train_lengths = self.experiment_config.experiment.softmax_train_info[str(seed)]
            for dim_idx, DIMENSION in enumerate(self.experiment_config.experiment.dimensions):
                CONTEXT_LEN = self.experiment_config.experiment.context_len_scale * DIMENSION
                emb_scale = self.experiment_config.experiment.emb_scale
                key_scale = self.experiment_config.experiment.key_scale
                    
                rng = random.PRNGKey(seed)
                rng, init_rng_softmax, train_rng_softmax, test_rng_softmax = random.split(rng,4)

                optim_sm = Optimizer(grad_clip=self.experiment_config.optim.grad_clip,
                            peak_lr=softmax_lrs[dim_idx] if self.special_lrs == None else self.special_lrs[dim_idx],
                            use_schedule=self.experiment_config.optim.use_schedule,
                            warmup_steps=self.experiment_config.optim.warmup_steps,
                            max_iters=self.experiment_config.optim.max_iters,
                            init_value=self.experiment_config.optim.init_value,
                            end_value=self.experiment_config.optim.end_value,
                            weight_decay=self.experiment_config.optim.weight_decay)
                
                optim_lin = Optimizer(grad_clip=self.experiment_config.optim.grad_clip,
                            peak_lr=self.experiment_config.experiment.linear_lr if self.linear_lrs == None else self.linear_lrs[dim_idx],
                            use_schedule=False,
                            warmup_steps=self.experiment_config.optim.warmup_steps,
                            max_iters=self.experiment_config.optim.max_iters,
                            init_value=self.experiment_config.optim.init_value,
                            end_value=self.experiment_config.optim.end_value,
                            weight_decay=self.experiment_config.optim.weight_decay)
        
                self.softmax_optimizer = optim_sm.get_optimizer()
                self.linear_optimizer = optim_lin.get_optimizer()

                # Create resp. Softmax Transformer and Training/Data Modules
                model_tf_softmax = self.get_linaerizable_softmaxTF(input_dim=int(emb_scale*DIMENSION),
                                                                    embed_dim=int(emb_scale*DIMENSION),
                                                                    key_size=int(key_scale*DIMENSION),
                                                                    seq_len=CONTEXT_LEN,
                                                                    data_dim=DIMENSION,
                                                                    linearize=False,
                                                                    linear_idx=-8)
                data_generator = self.get_data_generator(data_dim=DIMENSION, 
                                                            seq_len=CONTEXT_LEN,
                                                            embed_dim=int(emb_scale*DIMENSION))
                softmax_tf_train_module = Training(model=model_tf_softmax,
                                                   optimizer=self.softmax_optimizer,
                                                   data_generator=data_generator,
                                                   interpolate=False,
                                                   sensitivity_analysis=False,
                                                   batch_size=self.experiment_config.data.batch_size,
                                                   test_batch_size=self.experiment_config.data.test_batch_size)

                # Transformer Init & Training
                state_softmax_tf, rng = softmax_tf_train_module.get_init_state(init_rng_softmax, interpol_call=False)
                for epoch_idx in range(softmax_train_lengths[dim_idx]):
                    state_softmax_tf, train_rng_softmax, _, _, _ = softmax_tf_train_module.train_epoch(epoch=epoch_idx,
                                                                                                       state=state_softmax_tf,
                                                                                                       rng=train_rng_softmax,
                                                                                                       test_rng=test_rng_softmax,
                                                                                                       num_batches_train=100,
                                                                                                       interpolate=False)
                
                for LINEARIZATION_INDEX in range(self.model_config.num_layers + (1 if self.model_config.hybrid_first_block else 0)):

                    # Create resp. linear Transformer layer and Training/Helper Module
                    linear_layer = MultiHeadAttention(masked=True,
                                                        embed_dim = int(emb_scale*DIMENSION),
                                                        num_heads = self.model_config.num_heads,
                                                        use_softmax = False,
                                                        use_bias = False,
                                                        key_size = int(key_scale*DIMENSION),
                                                        initializer = self.model_config.initializer,
                                                        interpol=False,
                                                        seq_len=CONTEXT_LEN,
                                                        use_pe_kq=(True if self.model_config.use_pe_kq and LINEARIZATION_INDEX == 0 else False))
                    linearization_helper = LinearizationHelper(params_softmax=state_softmax_tf.params,
                                                                model_tf=model_tf_softmax,
                                                                linear_layer=linear_layer,
                                                                data_generator=data_generator,
                                                                num_batches_train=100,
                                                                layer_idx=LINEARIZATION_INDEX)
                    
                    # Linear layer Init & Training
                    rng, linear_init_rng, train_rng_linear, test_rng_linear = random.split(rng, 4)
                    init_params_rng, init_batch_rng = random.split(linear_init_rng)

                    (exmp_input,_),_ = data_generator.get_data(rng=init_batch_rng, batch_size=2)
                    _, (activations, _, _, _) = model_tf_softmax.apply({'params': state_softmax_tf.params}, exmp_input)

                    params_linear = linear_layer.init({'params': init_params_rng}, activations[LINEARIZATION_INDEX])['params']
                    state_linear = train_state.TrainState.create(apply_fn=linear_layer.apply, params=params_linear, tx=self.linear_optimizer)
                    
                    for _ in range(self.experiment_config.experiment.linear_train_len):
                        state_linear, train_rng_linear, _ = linearization_helper.train_epoch_lin(rng=train_rng_linear,
                                                                                                 state=state_linear,
                                                                                                 test_rng=test_rng_linear)
                    
                    # Get parameters and construct model with LINEARIZATION_INDEX softmax-layer switched for linear layer
                    rng, linearized_test_rng = random.split(rng)
                    qfm, kfm, vfm, pfm = tuple([state_linear.params[proj]['kernel'] for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']])
                    kernels = (qfm, kfm, vfm, pfm)
                    linearized_params = softmax_tf_train_module.replace_one_layer(state_softmax_tf.params,
                                                                                    interpol_kernels=kernels,
                                                                                    i=LINEARIZATION_INDEX,
                                                                                    hybrid=self.model_config.hybrid_first_block)
                    model_tf_linearized = self.get_linaerizable_softmaxTF(input_dim=int(emb_scale*DIMENSION),
                                                                            embed_dim=int(emb_scale*DIMENSION),
                                                                            key_size=int(key_scale*DIMENSION),
                                                                            seq_len=CONTEXT_LEN,
                                                                            data_dim=DIMENSION,
                                                                            linearize=True,
                                                                            linear_idx=LINEARIZATION_INDEX)
                    linearized_tf_train_module = Training(model=model_tf_linearized,
                                                            optimizer=self.softmax_optimizer,
                                                            data_generator=data_generator,
                                                            interpolate=False,
                                                            sensitivity_analysis=False,
                                                            batch_size=self.experiment_config.data.batch_size,
                                                            test_batch_size=self.experiment_config.data.test_batch_size)
                    
                    # Compute Test loss of linearized and original full-softmax model
                    test_loss_linearized = linearized_tf_train_module.fast_pure_test_computation(params=linearized_params, 
                                                                                                  test_rng=linearized_test_rng)
                    test_loss_tf = softmax_tf_train_module.fast_pure_test_computation(params=state_softmax_tf.params,
                                                                                              test_rng=linearized_test_rng)
                    
                    print(f'Model dim: {DIMENSION}. Normalized loss in layer {LINEARIZATION_INDEX}: {test_loss_linearized/test_loss_tf}')
                    results = results.at[seed_idx,dim_idx,LINEARIZATION_INDEX].set(test_loss_linearized/test_loss_tf)
                    
        return {'res_array' : results}

            
