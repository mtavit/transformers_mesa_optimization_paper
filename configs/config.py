from jax import nn as jax_nn
from ml_collections import config_dict
from typing import List

def get_experiment_config(seeds:List[int]) -> config_dict.ConfigDict:
    '''Returns the configuration for the model and the experiment'''
    experiment_config = config_dict.ConfigDict()
    experiment_config.seeds = seeds
    experiment_config.data = data_configurator() # Data Configurations
    experiment_config.optim = optim_configurator() # Optimizer Configurations
    experiment_config.experiment = experiment_configurator() # Specific Experiment Configurations
    return experiment_config

def get_model_config() -> config_dict.ConfigDict:
    '''Returns the model configuration for the given model description'''

    # you can easily either modify this config or just add new model_config functions

    def model_config():
        '''Fully Observed Data Constructed Transformer (Interpolation)'''
        model_config = config_dict.ConfigDict()
        model_config.use_emb=False      # use linear embedding layer (default: False when training on constructed tokens)
        model_config.use_pe_emb=False   # concatenate PE to embeddings (default: False when training on constructed tokens)
        model_config.use_pe_kq=False    # concatenate PE to K and Q in attention (default: False when training on constructed tokens)
        model_config.hybrid_first_block=False   # This adds an additional first layer to the model with possibly different settings. Also: If you want to use pe-kq, you need to set this to True
        model_config.hybrid_sm=False    # softmax or linear layer in hybrid layer (irrelevant if hybrid_first_block=False)
        model_config.num_hybrid_heads=0 # No. of heads in hybrid layer (irrelevant if hybrid_first_block=False)
        model_config.input_dim=40       # input dimension (of observed data)
        model_config.pe_dim=40          # positional encoding dimension (concatenated, if you want to change that, modify src/models/positional_encodings.py)
        model_config.out_dim=40         # output dimension
        model_config.data_dim=10        # data dimension (we introduce this field for one reason only: The constructed tokens are of form [0, xi, xi, xi-1]. But in the 1-layer TF trained on constructed tokens setting we mask away the second field for simplicity (doesn't affect computation) to obtain [0, 0, xi, xi-1].
        model_config.initializer=jax_nn.initializers.normal(stddev=0.0002)  # initializer for weights, in our experiments it wasn't necessary to adapt for deeper layers but instead use this as default
        model_config.use_layernorm=False    # use layernorm in transformer layers
        model_config.use_bias=False     # use bias in transformer layers (default = False)
        model_config.use_mlp=False      # use mlp in transformer layers
        model_config.masked=True        # causal masking in self-attention
        model_config.interpol=False     # activate for interpolation experiments (default = False)
        model_config.use_clip=True      # forward activation clipping
        model_config.clip_val=1.5       # clipping value [-clip_val, clip_val]
        model_config.mask_inputs=False  # see data_dim for explanation, used for a specific experiment
        model_config.use_mesa=False     # use mesa-self-attention instead of standard self-attention
        model_config.standard_backprop_mesa=False   # use mesa-self-attention with standard backpropagation
        model_config.normalization_mesa=False   # use mesa-self-attention with normalization
        model_config.num_layers=2       # number of transformer layers (without the optional first hybrid layer)
        model_config.num_heads=4        # number of heads in self-attention
        model_config.embed_dim=40       # embedding dimension
        model_config.key_size=20        # key size in self-attention
        model_config.seq_len=50         # sequence length of input sequences (unused field in TF, can be used for debugging)
        model_config.dim_feedforward_MLP=0    # dimension of hidden layer in MLP
        model_config.linear=True        # use linear self-attention
        model_config.linearize=False    # for linearization experiment
        model_config.linear_idx=False   # for linearization experiment
        model_config.use_schlagnorm=False   # use schlagnorm in transformer layers (normalize K,Q)
        model_config.schlagnorm_targets=False   # schlagnorm also for V
        model_config.use_schlagnorm_hyb=False   # use schlagnorm in hybrid layer (normalize K,Q)
        model_config.schlagnorm_targets_hyb=False   # schlagnorm also for V in hybrid layer
        return model_config

    return model_config()


def experiment_configurator() -> config_dict.ConfigDict:
    '''Returns the experiment configuration for the given experiment description'''
    experiment = config_dict.ConfigDict()
    experiment.interpolate = False     # activate for interpolation experiments
    experiment.sensitivity_analysis = False     # Unless necessary for specific experiment, leave this False
    experiment.probe_range = 7     # token-range for probing experiments
    return experiment


def optim_configurator() -> config_dict.ConfigDict:
    '''Returns the data configuration for the experiment'''                                                                                                                                                             # For nonlin-tf use 1e-3, for mesa 4e-4
    optim = config_dict.ConfigDict()
    optim.peak_lr = 3e-4    # Peak learning rate (or fixed if no scheduling)
    optim.grad_clip = 1     # Gradient clipping value
    optim.use_schedule = True   # Use learning rate scheduling
    optim.warmup_steps = 1000   # Warmup steps for learning rate scheduling
    optim.max_iters = 20000    # Maximum number of iterations for scheduling
    optim.init_value = 0    # Initial learning rate
    optim.end_value = 1e-5  # Final learning rate (at max_iters train steps)
    optim.weight_decay = 0.05   # Weight decay
    return optim

def data_configurator() -> config_dict.ConfigDict:
    '''Returns the data configuration for the experiment'''
    data = config_dict.ConfigDict()                                                                                                                          
    
    data.batch_size = 256   # Training batch size
    data.test_batch_size = 256  # Test batch size
    data.seq_len = 50   # Sequence length
    data.data_dim = 10  # Hidden Data dimension
    data.obs_dim = 10   # Observed data dimension
    data.noise_obs = 0  # Observation noise level, Only relevant for partially obserable sequences
    data.noise = 0.01   # Noise level
    data.eye_obs = True # Use identity matrix as observation matrix
    data.data_clip = 10 # Clip data to [-data_clip, data_clip] (relevant for contracting sequences)
    data.range = 1    # Range of initial values (U-distr.)
    data.construction = True    # Use constructed tokens
    data.type = 'seq_lin_constr_full' # seq_lin_constr_full for linear sequences with constructed tokens, seq_lin else
    data.embed_dim = 40     # Embedding dimension (for constructed tokens)
    return data