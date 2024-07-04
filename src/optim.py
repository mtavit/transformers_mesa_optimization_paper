'''Module for optimizer class'''

import optax

class Optimizer:
    '''Optimizer class for the model training.'''
    def __init__(self,
                 grad_clip: float = 1.0,
                 peak_lr: float = 3e-4,
                 use_schedule: bool = True,
                 warmup_steps: int = 500,
                 max_iters: int = 40000,
                 init_value: float = 0.0,
                 end_value: float = 1e-5,
                 weight_decay: float = 0.05):
        '''
        Initializes the optimizer with the specified parameters.
        Args:
            'grad_clip' (float): Gradient clipping value.
            'peak_lr' (float): Peak learning rate.
            'use_schedule' (bool): Flag whether to use the learning rate schedule.
            'warmup_steps' (int): Number of warmup steps.
            'max_iters' (int): Maximum number of training iterations.
            'init_value' (float): Initial learning rate value.
            'end_value' (float): Final learning rate value.
            'weight_decay' (float): Weight decay value.
        '''
        self.grad_clip = grad_clip
        self.peak_lr = peak_lr
        self.use_schedule = use_schedule
        self.warmup_steps = warmup_steps
        self.max_iters = max_iters
        self.init_value = init_value
        self.end_value = end_value
        self.weight_decay = weight_decay
    
    def get_optimizer(self):
        '''Returns the adamW-optimizer chain with the specified parameters.'''
        lr_schedule = optax.warmup_cosine_decay_schedule(init_value=self.init_value,
                                                         peak_value=self.peak_lr,
                                                         warmup_steps=self.warmup_steps,
                                                         decay_steps=self.max_iters,
                                                         end_value=self.end_value)
        if self.use_schedule:
            return optax.chain(optax.clip(self.grad_clip),
                               optax.adamw(lr_schedule, weight_decay=self.weight_decay))
        else:
            return optax.chain(optax.clip(self.grad_clip),
                               optax.adamw(self.peak_lr, weight_decay=self.weight_decay))
