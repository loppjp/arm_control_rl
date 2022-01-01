from typing import Tuple

import numpy as np 

class NormalDecayNoise:
    """
    Generate normally distributed noise around 0.0
    Allow calling code to specify the maximum 
    and minimum scaled values. Values will linearly
    decreace from maximum to minimum based on a provided
    scheduled number of decay steps. Beyond end of decay steps
    The value will plateu at the min value.
    """

    def __init__(
        self,
        size:Tuple,
        max_noise:float=0.5,
        min_noise:float=0.1,
        decay_steps:int=10000
    ):

        self.size = size
        self.max_noise = max_noise
        self.min_noise = min_noise
        self.decay_steps = decay_steps
        self.scale = self.max_noise

    def __call__(self, step):
        """
        reduce noise over time
        """

        self.scale=self.max_noise - \
                (self.max_noise-self.min_noise) * \
                    min(self.decay_steps, step) / self.decay_steps

        return np.random.normal(
            loc=0,
            scale=self.scale,
            size=self.size
        )
