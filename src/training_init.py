
'''Module for initializing training components'''

from ml_collections import config_dict

from src.models.full_fledged_transformer import FullTransformerModel
from src.optim import Optimizer
from src.data.seq.linear_seq import LinearSequenceDataGenerator
from src.data.seq.fixed_w_linear_seq import FixedWDataGenerator
from src.data.seq.nonlinear_seq import NonlinearSequenceDataGenerator
from src.data.seq.contracting_seq import ContractingSequenceDataGenerator
from src.data.constr_data import ConstructedFullSeqGenerator, ConstructedPartObsGenerator
from src.data.icl.icl_single import ICLSingleGenerator
from src.data.icl.icl_multi import ICLMultiGenerator
from src.train import Training

class TrainingInitializer:
    '''Class for initializing training components'''
    def __init__(self, 
                 model_config: config_dict.ConfigDict,
                 experiment_config: config_dict.ConfigDict):
        '''
        Initializes the training components
        Args:
            model_config: Configuration for the model
            experiment_config: Configuration for the experiment
        '''
        
        self.model_config = model_config
        self.experiment_config = experiment_config

    def get_tf_model(self):
        '''Returns the transformer model based on the configuration'''
        return FullTransformerModel(use_emb=self.model_config.use_emb,
                                    use_pe_emb=self.model_config.use_pe_emb,
                                    use_pe_kq=self.model_config.use_pe_kq,
                                    hybrid_first_block=self.model_config.hybrid_first_block,
                                    hybrid_sm=self.model_config.hybrid_sm,
                                    num_hybrid_heads=self.model_config.num_hybrid_heads,
                                    input_dim=self.model_config.input_dim,
                                    pe_dim=self.model_config.pe_dim,
                                    out_dim=self.model_config.out_dim,
                                    data_dim=self.experiment_config.data.data_dim,
                                    initializer=self.model_config.initializer,
                                    use_layernorm=self.model_config.use_layernorm,
                                    use_bias=self.model_config.use_bias,
                                    use_mlp=self.model_config.use_mlp,
                                    masked=self.model_config.masked,
                                    interpol=self.model_config.interpol,
                                    use_clip=self.model_config.use_clip,
                                    clip=self.model_config.clip_val,
                                    mask_inputs=self.model_config.mask_inputs,
                                    use_mesa=self.model_config.use_mesa,
                                    standard_backprop_mesa=self.model_config.standard_backprop_mesa,
                                    normalization_mesa=self.model_config.normalization_mesa,
                                    num_layers=self.model_config.num_layers,
                                    num_heads=self.model_config.num_heads,
                                    embed_dim=self.model_config.embed_dim,
                                    key_size=self.model_config.key_size,
                                    seq_len=self.experiment_config.data.seq_len,
                                    dim_feedforward_MLP=self.model_config.dim_feedforward_MLP,
                                    linear=self.model_config.linear,
                                    linearize=self.model_config.linearize,
                                    linear_idx=self.model_config.linear_idx,
                                    use_schlagnorm_hyb=self.model_config.use_schlagnorm_hyb,
                                    schlagnorm_targets_hyb=self.model_config.schlagnorm_targets_hyb,
                                    use_schlagnorm=self.model_config.use_schlagnorm,
                                    schlagnorm_targets=self.model_config.schlagnorm_targets)
    
    def get_optimizer(self):
        '''Returns the optimizer based on the configuration'''
        return Optimizer(grad_clip=self.experiment_config.optim.grad_clip,
                            peak_lr=self.experiment_config.optim.peak_lr,
                            use_schedule=self.experiment_config.optim.use_schedule,
                            warmup_steps=self.experiment_config.optim.warmup_steps,
                            max_iters=self.experiment_config.optim.max_iters,
                            init_value=self.experiment_config.optim.init_value,
                            end_value=self.experiment_config.optim.end_value,
                            weight_decay=self.experiment_config.optim.weight_decay).get_optimizer()
    
    def get_data_generator(self, type:str):
        '''Returns the data generator based on the configuration'''
        if type == 'seq_lin':
            return LinearSequenceDataGenerator(seq_len=self.experiment_config.data.seq_len,
                                                data_dim=self.experiment_config.data.data_dim,
                                                obs_dim=self.experiment_config.data.obs_dim,
                                                range=self.experiment_config.data.range,
                                                noise=self.experiment_config.data.noise,
                                                noise_obs=self.experiment_config.data.noise_obs,
                                                data_clip=self.experiment_config.data.data_clip,
                                                eye_obs=self.experiment_config.data.eye_obs)
        elif type == 'fixed':
            return FixedWDataGenerator(seq_len=self.experiment_config.data.seq_len,
                                        data_dim=self.experiment_config.data.data_dim,
                                        obs_dim=self.experiment_config.data.obs_dim,
                                        range=self.experiment_config.data.range,
                                        noise=self.experiment_config.data.noise,
                                        noise_obs=self.experiment_config.data.noise_obs,
                                        data_clip=self.experiment_config.data.data_clip,
                                        eye_obs=self.experiment_config.data.eye_obs)
        elif type == 'seq_nonlin':
            return NonlinearSequenceDataGenerator(A_mat=self.experiment_config.data.A_mat,
                                                    B_mat=self.experiment_config.data.B_mat,
                                                    seq_len=self.experiment_config.data.seq_len,
                                                    data_dim=self.experiment_config.data.data_dim,
                                                    range=self.experiment_config.data.range,
                                                    noise=self.experiment_config.data.noise)
        elif type == 'contracting':
            return ContractingSequenceDataGenerator(seq_len=self.experiment_config.data.seq_len,
                                                    data_dim=self.experiment_config.data.data_dim,
                                                    range=self.experiment_config.data.range,
                                                    noise=self.experiment_config.data.noise,
                                                    noise_obs=self.experiment_config.data.noise_obs,
                                                    data_clip=self.experiment_config.data.data_clip,
                                                    obs_dim=self.experiment_config.data.obs_dim,
                                                    eye_obs=self.experiment_config.data.eye_obs,
                                                    min_eig=self.experiment_config.data.min_eig,
                                                    max_eig=self.experiment_config.data.max_eig)
        elif type == 'seq_lin_constr_full':
            return ConstructedFullSeqGenerator(data_generator=self.get_data_generator('seq_lin'),
                                                embed_dim=self.experiment_config.data.embed_dim)
        elif type == 'seq_lin_constr_part':
            return ConstructedPartObsGenerator(data_generator=self.get_data_generator('seq_lin'),
                                                embed_dim=self.experiment_config.data.embed_dim)
        elif type == 'seq_lin_constr_contracting':
            return ConstructedFullSeqGenerator(data_generator=self.get_data_generator('contracting'),
                                                embed_dim=self.experiment_config.data.embed_dim)
        elif type == 'fixed_constr':
            return ConstructedFullSeqGenerator(data_generator=self.get_data_generator('fixed'),
                                                embed_dim=self.experiment_config.data.embed_dim)
        elif type == 'icl_single':
            return ICLSingleGenerator(seq_len=self.experiment_config.data.seq_len,
                                        data_dim=self.experiment_config.data.data_dim,
                                        range=self.experiment_config.data.range,
                                        noise=self.experiment_config.data.noise)
        elif type == 'icl_multi':
            return ICLMultiGenerator(seq_len_task1=self.experiment_config.data.seq_len_task1,
                                        seq_len_task2=self.experiment_config.data.seq_len_task2,
                                        data_dim=self.experiment_config.data.data_dim,
                                        range=self.experiment_config.data.range,
                                        noise=self.experiment_config.data.noise)
        else:
            raise ValueError('Data type not recognized')
        
    def get_train_module(self, model, optimizer, data_generator):
            '''Returns the training module based on the configuration'''
            return Training(model=model,
                            optimizer=optimizer,
                            data_generator=data_generator,
                            interpolate=self.experiment_config.experiment.interpolate,
                            sensitivity_analysis=self.experiment_config.experiment.sensitivity_analysis,
                            batch_size=self.experiment_config.data.batch_size,
                            test_batch_size=self.experiment_config.data.test_batch_size,)
    
    def run(self):
        '''Returns the components'''
        model = self.get_tf_model()
        optimizer = self.get_optimizer()
        data_generator = self.get_data_generator(self.experiment_config.data.type)
        train_module = self.get_train_module(model, optimizer, data_generator)
        
        return (model, optimizer, data_generator, train_module)