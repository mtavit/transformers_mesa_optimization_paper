from jax import random, nn as jax_nn
from ml_collections import config_dict
from typing import List

def get_experiment_config(experiment:str, seeds:List[int]) -> config_dict.ConfigDict:
    '''Returns the configuration for the model and the experiment'''
    experiment_config = config_dict.ConfigDict()
    experiment_config.seeds = seeds
    experiment_config.data = data_configurator(experiment) # Data Configurations
    experiment_config.optim = optim_configurator(experiment) # Optimizer Configurations
    experiment_config.experiment = experiment_configurator(experiment) # Specific Experiment Configurations
    return experiment_config

def get_model_config(model_desc:str, print_specs:bool=True) -> config_dict.ConfigDict:
    '''Returns the model configuration for the given model description'''
    
    model, layers, heads, embed_dim, seq_len, key_size, pe_kq, pe_emb, laynorm, mlp, dim_mlp, linear, clip, clip_val, schlagnorm, schlagnorm_t, schlagnorm_hyb, schlagnorm_t_hyb = model_desc.split('.')
    if print_specs:
        print('model: ', model, ' layers:', layers,' heads: ', heads,' embed_dim: ', embed_dim,' seq_len: ', seq_len,' key_size: ', key_size,' pe_kq: ', pe_kq,' pe_emb: ', pe_emb,' laynorm: ', laynorm,' mlp: ', mlp,' dim_mlp: ', dim_mlp,' linear: ', linear,' clip: ', clip,' clip_val: ', clip_val)
    layers = int(layers)
    heads = int(heads)
    embed_dim = int(embed_dim)
    seq_len = int(seq_len)
    key_size = int(key_size)
    pe_kq = pe_kq == 'True'
    pe_emb = pe_emb == 'True'
    laynorm = laynorm == 'True'
    mlp = mlp == 'True'
    dim_mlp = int(dim_mlp)
    linear = linear == 'True'
    clip = clip == 'True'
    clip_val = float(clip_val)
    schlagnorm = schlagnorm == 'True'
    schlagnorm_hyb = schlagnorm_hyb == 'True'
    schlagnorm_t = schlagnorm_t == 'True'
    schlagnorm_t_hyb = schlagnorm_t_hyb == 'True'

    def model_conf1():
        '''Fully Observed Data Full Fledged Transformer'''
        model_config = config_dict.ConfigDict()
        model_config.use_emb=True
        model_config.use_pe_emb=pe_emb
        model_config.use_pe_kq=pe_kq
        model_config.hybrid_first_block=True
        model_config.hybrid_sm=True
        model_config.num_hybrid_heads=heads
        model_config.input_dim=10
        model_config.pe_dim=embed_dim
        model_config.out_dim=10
        model_config.data_dim=10
        model_config.initializer=jax_nn.initializers.normal(stddev=0.05)
        model_config.use_layernorm=laynorm
        model_config.use_bias=False
        model_config.use_mlp=mlp
        model_config.masked=True
        model_config.interpol=False
        model_config.use_clip=False
        model_config.clip_val=1.5
        model_config.mask_inputs=False
        model_config.use_mesa=False
        model_config.standard_backprop_mesa=False
        model_config.normalization_mesa=False
        model_config.num_layers=layers
        model_config.num_heads=heads
        model_config.embed_dim=embed_dim
        model_config.key_size=key_size
        model_config.seq_len=seq_len
        model_config.dim_feedforward_MLP=dim_mlp
        model_config.linear=linear
        model_config.linearize=False
        model_config.linear_idx=False
        model_config.use_schlagnorm=schlagnorm
        model_config.schlagnorm_targets=schlagnorm_t
        model_config.use_schlagnorm_hyb=schlagnorm_hyb
        model_config.schlagnorm_targets_hyb=schlagnorm_t_hyb
        return model_config
    
    def model_just_mesa():
        '''Fully Observed Data Pure Mesa Transformer'''
        model_config = config_dict.ConfigDict()
        model_config.use_emb=True
        model_config.use_pe_emb=pe_emb
        model_config.use_pe_kq=pe_kq
        model_config.hybrid_first_block=True
        model_config.hybrid_sm=False
        model_config.hybrid_mesa=True
        model_config.num_hybrid_heads=heads
        model_config.input_dim=10
        model_config.pe_dim=embed_dim
        model_config.out_dim=10
        model_config.data_dim=10
        model_config.initializer=jax_nn.initializers.normal(stddev=0.05)
        model_config.use_layernorm=laynorm
        model_config.use_bias=False
        model_config.use_mlp=mlp
        model_config.masked=True
        model_config.interpol=False
        model_config.use_clip=False
        model_config.clip_val=1.5
        model_config.mask_inputs=False
        model_config.use_mesa=True
        model_config.standard_backprop_mesa=False
        model_config.normalization_mesa=False
        model_config.num_layers=layers
        model_config.num_heads=heads
        model_config.embed_dim=embed_dim
        model_config.key_size=key_size
        model_config.seq_len=seq_len
        model_config.dim_feedforward_MLP=dim_mlp
        model_config.linear=linear
        model_config.linearize=False
        model_config.linear_idx=False
        model_config.use_schlagnorm=schlagnorm
        model_config.schlagnorm_targets=schlagnorm_t
        model_config.use_schlagnorm_hyb=schlagnorm_hyb
        model_config.schlagnorm_targets_hyb=schlagnorm_t_hyb
        return model_config

    def model_conf2():
        '''Partially Observed Data Full Fledged Transformer'''
        model_config = config_dict.ConfigDict()
        model_config.use_emb=True
        model_config.use_pe_emb=pe_emb
        model_config.use_pe_kq=pe_kq
        model_config.hybrid_first_block=True
        model_config.hybrid_sm=True
        model_config.num_hybrid_heads=heads
        model_config.input_dim=5
        model_config.pe_dim=embed_dim
        model_config.out_dim=5
        model_config.data_dim=5
        model_config.initializer=jax_nn.initializers.normal(stddev=0.05)
        model_config.use_layernorm=laynorm
        model_config.use_bias=False
        model_config.use_mlp=mlp
        model_config.masked=True
        model_config.interpol=False
        model_config.use_clip=False
        model_config.clip_val=1.5
        model_config.mask_inputs=False
        model_config.use_mesa=False
        model_config.standard_backprop_mesa=False
        model_config.normalization_mesa=False
        model_config.num_layers=layers
        model_config.num_heads=heads
        model_config.embed_dim=embed_dim
        model_config.key_size=key_size
        model_config.seq_len=seq_len
        model_config.dim_feedforward_MLP=dim_mlp
        model_config.linear=linear
        model_config.linearize=False
        model_config.linear_idx=False
        model_config.use_schlagnorm=False
        model_config.schlagnorm_targets=False
        model_config.use_schlagnorm_hyb=False
        model_config.schlagnorm_targets_hyb=False
        return model_config

    def model_conf3():
        '''Fully Observed Data Constructed Transformer (Interpolation)'''
        model_config = config_dict.ConfigDict()
        model_config.use_emb=False
        model_config.use_pe_emb=False
        model_config.use_pe_kq=False
        model_config.hybrid_first_block=False
        model_config.hybrid_sm=False
        model_config.num_hybrid_heads=0
        model_config.input_dim=40
        model_config.pe_dim=embed_dim
        model_config.out_dim=40
        model_config.data_dim=10
        model_config.initializer=jax_nn.initializers.normal(stddev=0.0002)
        model_config.use_layernorm=False
        model_config.use_bias=False
        model_config.use_mlp=False
        model_config.masked=True
        model_config.interpol=True
        model_config.use_clip=clip
        model_config.clip_val=clip_val
        model_config.mask_inputs=(layers==1)
        model_config.use_mesa=False
        model_config.standard_backprop_mesa=False
        model_config.normalization_mesa=False
        model_config.num_layers=layers
        model_config.num_heads=heads
        model_config.embed_dim=embed_dim
        model_config.key_size=key_size
        model_config.seq_len=seq_len
        model_config.dim_feedforward_MLP=0
        model_config.linear=linear
        model_config.linearize=False
        model_config.linear_idx=False
        model_config.use_schlagnorm=False
        model_config.schlagnorm_targets=False
        model_config.use_schlagnorm_hyb=False
        model_config.schlagnorm_targets_hyb=False
        return model_config
    
    def mesa_fully_obs_constr():
        '''Fully Observed Data Constructed MESA-Transformer'''
        model_config = config_dict.ConfigDict()
        model_config.use_emb=False
        model_config.use_pe_emb=False
        model_config.use_pe_kq=False
        model_config.hybrid_first_block=False
        model_config.hybrid_sm=False
        model_config.num_hybrid_heads=0
        model_config.input_dim=40
        model_config.pe_dim=embed_dim
        model_config.out_dim=40
        model_config.data_dim=10
        model_config.initializer=jax_nn.initializers.normal(stddev=0.0002)
        model_config.use_layernorm=False
        model_config.use_bias=False
        model_config.use_mlp=False
        model_config.masked=True
        model_config.interpol=True
        model_config.use_clip=clip
        model_config.clip_val=clip_val
        model_config.mask_inputs=(layers==1)
        model_config.use_mesa=True
        model_config.standard_backprop_mesa=False
        model_config.normalization_mesa=False
        model_config.num_layers=layers
        model_config.num_heads=heads
        model_config.embed_dim=embed_dim
        model_config.key_size=key_size
        model_config.seq_len=seq_len
        model_config.dim_feedforward_MLP=0
        model_config.linear=linear
        model_config.linearize=False
        model_config.linear_idx=False
        model_config.use_schlagnorm=False
        model_config.schlagnorm_targets=False
        model_config.use_schlagnorm_hyb=False
        model_config.schlagnorm_targets_hyb=False
        return model_config
    
    def mesa_fully_obs():
        '''Fully Observed Data full-fledged-(hybrid)-MESA-Transformer'''
        model_config = config_dict.ConfigDict()
        model_config.use_emb=True
        model_config.use_pe_emb=pe_emb
        model_config.use_pe_kq=pe_kq
        model_config.hybrid_first_block=True
        model_config.hybrid_sm=True
        model_config.num_hybrid_heads=heads
        model_config.input_dim=10
        model_config.pe_dim=embed_dim
        model_config.out_dim=10
        model_config.data_dim=10
        model_config.initializer=jax_nn.initializers.normal(stddev=0.05)
        model_config.use_layernorm=laynorm
        model_config.use_bias=False
        model_config.use_mlp=mlp
        model_config.masked=True
        model_config.interpol=False
        model_config.use_clip=clip
        model_config.clip_val=clip_val
        model_config.mask_inputs=False
        model_config.use_mesa=True
        model_config.standard_backprop_mesa=False
        model_config.normalization_mesa=False
        model_config.num_layers=layers
        model_config.num_heads=heads
        model_config.embed_dim=embed_dim
        model_config.key_size=key_size
        model_config.seq_len=seq_len
        model_config.dim_feedforward_MLP=dim_mlp
        model_config.linear=linear
        model_config.linearize=False
        model_config.linear_idx=False
        model_config.use_schlagnorm=schlagnorm
        model_config.schlagnorm_targets=schlagnorm_t
        model_config.use_schlagnorm_hyb=schlagnorm_hyb
        model_config.schlagnorm_targets_hyb=schlagnorm_t_hyb
        return model_config
    
    def mesa_part_obs():
        '''Partially Observed Data full-fledged-(hybrid)-MESA-Transformer'''
        model_config = config_dict.ConfigDict()
        model_config.use_emb=True
        model_config.use_pe_emb=pe_emb
        model_config.use_pe_kq=pe_kq
        model_config.hybrid_first_block=True
        model_config.hybrid_sm=True
        model_config.num_hybrid_heads=heads
        model_config.input_dim=5
        model_config.pe_dim=embed_dim
        model_config.out_dim=5
        model_config.data_dim=5
        model_config.initializer=jax_nn.initializers.normal(stddev=0.05)
        model_config.use_layernorm=laynorm
        model_config.use_bias=False
        model_config.use_mlp=mlp
        model_config.masked=True
        model_config.interpol=False
        model_config.use_clip=clip
        model_config.clip_val=clip_val
        model_config.mask_inputs=False
        model_config.use_mesa=True
        model_config.standard_backprop_mesa=False
        model_config.normalization_mesa=False
        model_config.num_layers=layers
        model_config.num_heads=heads
        model_config.embed_dim=embed_dim
        model_config.key_size=key_size
        model_config.seq_len=seq_len
        model_config.dim_feedforward_MLP=dim_mlp
        model_config.linear=linear
        model_config.linearize=False
        model_config.linear_idx=False
        model_config.use_schlagnorm=False
        model_config.schlagnorm_targets=False
        model_config.use_schlagnorm_hyb=False
        model_config.schlagnorm_targets_hyb=False
        return model_config

    return {
        'fully_obs_full_fledged_transformer' : model_conf1(),
        'part_obs_full_fledged_transformer'  : model_conf2(),
        'just_mesa'                          : model_just_mesa(),
        'fully_obs_constructed_transformer'  : model_conf3(),
        'nonlinear_full_fledged_transformer' : model_conf1(),
        'fully_obs_constructed_mesa'         : mesa_fully_obs_constr(),
        'fully_obs_full_fledged_mesa'        : mesa_fully_obs(),
        'part_obs_full_fledged_mesa'         : mesa_part_obs(),
    }[model]


def experiment_configurator(exp:str) -> config_dict.ConfigDict:
    '''Returns the experiment configuration for the given experiment description'''
    experiment = config_dict.ConfigDict()
    experiment.interpolate = False
    experiment.sensitivity_analysis = False
    experiment.batch_norm_threshold = 2
    experiment.inv_probe_lambda = 0.001
    experiment.probe_range = 7
    if exp == 'interpol_one':
        experiment.init_train_len = 25
        experiment.interpol_retrain_len = 35
        experiment.interpolate = True
    elif exp == 'interpol_six':
        experiment.init_train_len = 35
        experiment.interpol_retrain_len = 45
        experiment.interpolate = True
    elif exp == 'icl':
        experiment.experiment_length = 68
        experiment.full_experiment_length = 224
    elif exp == 'nonlin':
        experiment.train_len = 400
    elif exp == 'sensitivity':
        experiment.train_len = 30
        experiment.analysis_range = 5
        experiment.sensitivity_analysis = True
    elif exp == 'partobs':
        experiment.probe_range = 10
    elif exp[:9] == 'linearize':
        experiment.layer_indices = range(7)
        experiment.dimensions = [4,6,10,20,40,60]
        experiment.context_len_scale = 4
        experiment.emb_scale = 4
        experiment.key_scale = 2
        experiment.linear_train_len = 150
        experiment.linear_lr = 1e-3
        experiment.softmax_train_info = {'1':     ([4e-3,4e-3,4e-3,9e-3,7e-3,4e-3], [50,50,50,80,35,25]),
                                         '11':    ([4e-3,4e-3,4e-3,9e-3,6e-3,4e-3], [50,50,50,80,15,30]),
                                         '111':   ([4e-3,4e-3,4e-3,9e-3,7e-3,3e-3], [30,50,50,80,15,30]),
                                         '1111':  ([4e-3,4e-3,4e-3,9e-3,7e-3,3e-3], [50,50,50,80,20,30]),
                                         '11111': ([4e-3,4e-3,4e-3,8e-3,7e-3,3e-3], [50,50,50,80,45,30])}
    return experiment


def optim_configurator(experiment:str) -> config_dict.ConfigDict:
    '''Returns the data configuration for the experiment'''                                                                                                                                                             # For nonlin-tf use 1e-3, for mesa 4e-4
    optim = config_dict.ConfigDict()
    learning_rates =    {'interpol_one':2e-4,       'interpol_six':1e-4,         'sensitivity':7e-4,          'partobs':4e-4,          'icl':8e-4,         'probing_full':8e-4,          'probing_constr':1e-4,         'nonlin':1e-3,          'linearize_six':7e-4,       'contracting':1e-4}                                                  
    grad_clips =        {'interpol_one':1,          'interpol_six':1,            'sensitivity':1,             'partobs':1,             'icl':1,            'probing_full':1,             'probing_constr':1,            'nonlin':1,             'linearize_six':1,          'contracting':1}
    use_schedules =     {'interpol_one':False,      'interpol_six':False,        'sensitivity':True,          'partobs':True,          'icl':True,         'probing_full':True,          'probing_constr':False,        'nonlin':True,          'linearize_six':True,       'contracting':False}
    warmup_steps =      {'interpol_one':1000,       'interpol_six':1000,         'sensitivity':1000,          'partobs':1000,          'icl':1000,         'probing_full':1000,          'probing_constr':1000,         'nonlin':1000,          'linearize_six':1000,       'contracting':1000}
    max_iters =         {'interpol_one':10000,      'interpol_six':20000,        'sensitivity':30000,         'partobs':30000,         'icl':30000,        'probing_full':30000,         'probing_constr':25000,        'nonlin':50000,         'linearize_six':30000,      'contracting':20000}
    init_values =       {'interpol_one':0,          'interpol_six':0,            'sensitivity':0,             'partobs':0,             'icl':0,            'probing_full':0,             'probing_constr':0,            'nonlin':0,             'linearize_six':0,          'contracting':0}
    end_values =        {'interpol_one':1e-5,       'interpol_six':1e-5,         'sensitivity':1e-5,          'partobs':1e-5,          'icl':1e-5,         'probing_full':1e-5,          'probing_constr':1e-5,         'nonlin':1e-5,          'linearize_six':1e-5,       'contracting':1e-5}
    weight_decays =     {'interpol_one':0.05,       'interpol_six':0.1,          'sensitivity':0.05,          'partobs':0.05,          'icl':0.05,         'probing_full':0.05,          'probing_constr':0.1,          'nonlin':0.05,          'linearize_six':0.05,       'contracting':0.1}
    
    if experiment == 'fixed_w_full':
        experiment = 'probing_full'
    elif experiment == 'fixed_w_constr':
        experiment = 'probing_constr'
    elif experiment == 'linearize_full':
        experiment = 'probing_full'
    optim.peak_lr = learning_rates[experiment]
    
    optim.grad_clip = grad_clips[experiment]
    optim.use_schedule = use_schedules[experiment]
    optim.warmup_steps = warmup_steps[experiment]
    optim.max_iters = max_iters[experiment]
    optim.init_value = init_values[experiment]
    optim.end_value = end_values[experiment]
    optim.weight_decay = weight_decays[experiment]

    return optim

def data_configurator(experiment:str) -> config_dict.ConfigDict:
    '''Returns the data configuration for the experiment'''
    data = config_dict.ConfigDict()                                                                                                                          
    batch_sizes =       {'interpol_one':256,        'interpol_six':512,            'sensitivity':256,           'partobs':256,           'icl':128,          'probing_full':256,           'probing_constr':256,          'nonlin':256,         'contracting':256,      'contracting_full':256,    'fixed_w_constr':512}                                       
    test_batch_sizes =  {'interpol_one':256,        'interpol_six':256,            'sensitivity':256,           'partobs':256,           'icl':128,          'probing_full':256,           'probing_constr':256,          'nonlin':256,         'contracting':256,      'contracting_full':256,     'fixed_w_constr':256}
    seq_lens =          {'interpol_one':50,         'interpol_six':50,             'sensitivity':50,            'partobs':50,            'icl':224,          'probing_full':50,            'probing_constr':50,           'nonlin':50,          'contracting':50,       'contracting_full':50,      'fixed_w_constr':50}
    data_dims =         {'interpol_one':10,         'interpol_six':10,             'sensitivity':10,            'partobs':15,            'icl':10,           'probing_full':10,            'probing_constr':10,           'nonlin':10,          'contracting':10,       'contracting_full':10,      'fixed_w_constr':10}	
    obs_dims =          {'interpol_one':10,         'interpol_six':10,             'sensitivity':10,            'partobs':5,             'icl':10,           'probing_full':10,            'probing_constr':10,           'nonlin':10,          'contracting':10,       'contracting_full':10,      'fixed_w_constr':10}
    noises_obs =        {'interpol_one':0.00,       'interpol_six':0.00,           'sensitivity':0.00,          'partobs':0.01,          'icl':0.00,         'probing_full':0.00,          'probing_constr':0.00,         'nonlin':0.00,        'contracting':0.00,     'contracting_full':0.00,    'fixed_w_constr':0.00}
    noises =            {'interpol_one':0.01,       'interpol_six':0.01,           'sensitivity':0.01,          'partobs':0.01,          'icl':0.01,         'probing_full':0.01,          'probing_constr':0.01,         'nonlin':0.01,        'contracting':0.01,     'contracting_full':0.01,    'fixed_w_constr':0.01}
    eye_obs =           {'interpol_one':True,       'interpol_six':True,           'sensitivity':True,          'partobs':False,         'icl':True,         'probing_full':True,          'probing_constr':True,         'nonlin':True,        'contracting':True,     'contracting_full':True,    'fixed_w_constr':True}
    data_clips =        {'interpol_one':10,         'interpol_six':10,             'sensitivity':10,            'partobs':10,            'icl':10,           'probing_full':10,            'probing_constr':10,           'nonlin':10,          'contracting':2,        'contracting_full':2,       'fixed_w_constr':10}
    ranges =            {'interpol_one':1,          'interpol_six':1,              'sensitivity':1,             'partobs':1,             'icl':1,            'probing_full':1,             'probing_constr':1,            'nonlin':0.25,        'contracting':1,        'contracting_full':1,       'fixed_w_constr':1}    
    constructions =     {'interpol_one':True,       'interpol_six':True,           'sensitivity':False,         'partobs':True,          'icl':False,        'probing_full':False,         'probing_constr':True,         'nonlin':False,       'contracting':True,     'contracting_full':False,   'fixed_w_constr':True}
    embed_dims =        {'interpol_one':40,         'interpol_six':40,             'sensitivity':40,            'partobs':80,            'icl':40,           'probing_full':40,            'probing_constr':40,           'nonlin':50,          'contracting':40,       'contracting_full':40,      'fixed_w_constr':40}
    
    types = {'interpol_one':'seq_lin_constr_full',
             'interpol_six':'seq_lin_constr_full',       
             'sensitivity':'seq_lin',     
             'partobs':'seq_lin',     
             'icl':'seq_lin',    
             'probing_full':'seq_lin',     
             'probing_constr':'seq_lin_constr_full',    
             'nonlin':'seq_nonlin',
             'contracting':'seq_lin_constr_contracting',
             'contracting_full':'contracting',
             'fixed_w_full':'fixed',
             'fixed_w_constr':'fixed_constr'}
    
    data.batch_size = 64 if experiment in ['linearize_six', 'linearize_one', 'linearize_full'] else batch_sizes[experiment]
    data.test_batch_size = 64 if experiment in ['linearize_six', 'linearize_one', 'linearize_full'] else test_batch_sizes[experiment]
    
    if experiment == 'linearize_six' or experiment == 'linearize_one':
        experiment = 'probing_constr'
    elif experiment == 'linearize_full':
        experiment = 'probing_full'

    data.seq_len = seq_lens[experiment]
    data.data_dim = data_dims[experiment]
    data.obs_dim = obs_dims[experiment]
    data.noise_obs = noises_obs[experiment]
    data.noise = noises[experiment]
    data.eye_obs = eye_obs[experiment]
    data.data_clip = data_clips[experiment]
    data.range = ranges[experiment]
    data.construction = constructions[experiment]
    data.type = types[experiment]
    data.embed_dim = embed_dims[experiment]

    # Min/Max Eigenvalues only relevant for contracting sequences:
    data.min_eig = 0.3
    data.max_eig = 0.9

    data.A_mat = 1.1*random.normal(random.PRNGKey(421), shape = (data_dims[experiment], 4*data_dims[experiment]))
    data.B_mat = 1.1*random.normal(random.PRNGKey(422), shape = (4*data_dims[experiment], data_dims[experiment]))
    return data