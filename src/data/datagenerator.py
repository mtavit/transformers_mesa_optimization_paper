
'''DataGenerator Abstract Base Class'''

import abc
from jax import random, numpy as jnp
from typing import Tuple

class DataGenerator(metaclass=abc.ABCMeta):
    '''Abstract Base Class for DataGenerator'''

    @abc.abstractmethod
    def get_data(self, 
                 rng: random.PRNGKey, 
                 batch_size: int, 
                 **kwargs) -> Tuple[Tuple[jnp.ndarray]]:
        '''Abstract method to get data batch'''
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_data_info(self):
        '''Abstract method to get data info'''
        raise NotImplementedError