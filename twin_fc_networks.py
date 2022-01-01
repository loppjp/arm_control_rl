from typing import List, Callable

import torch
from torch  import nn

from fc_network import Network

class DerivedNetwork(Network):

    def seed_func(self, seed):
        pass

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Twin(torch.nn.Module):

    def __init__(self, 
        input_size:int=1, 
        output_size:int=1, 
        cat_size:int=0,
        #hidden_layers=[64,32,32,16],
        #hidden_layers=[32,32,16],
        #hidden_layers=[32,32],
        #hidden_layers=[64,64],
        #hidden_layers=[128,64,64,32],
        #hidden_layers=[256,256,128,128],
        #hidden_layers=[64,64],
        #hidden_layers=[128,128],
        hidden_layers=[256,256],
        #hidden_layers=[512,512],
        #hidden_layers=[128,128,128],
        #hidden_layers=[256,256,256,256],
        #hidden_layers:List[int]=[256,256,128,128,64,64,32],
        seed:int=1234,
        #internal_activation_fn:Callable=torch.nn.LeakyReLU,
        internal_activation_fn:Callable=nn.ReLU,
        #internal_activation_fn=F.linear,
        #internal_activation_fn=None,
        #output_activation_fn=torch.tanh,
        output_activation_fn:Callable=None,
        #output_activation_fn=F.linear,
        #output_activation_fn=F.leaky_relu,
        #output_activation_fn=None,
        batch_size:int = 1,
        layer_norm:bool=False
    ):
        """
        Instantiate Neural Network to approximate action value function

        Arguments:

            input_size (int): Demension of input, usually state vector
            output_size (int): Demension of output, multiple for actions,
                               1 for state value estimation
            seed (int): Random seed for reproducability 
        """
        super().__init__()
        assert(len(hidden_layers) > 0)
        self.seed = torch.manual_seed(seed)

        self.a = DerivedNetwork(
            input_size=input_size,
            output_size=output_size,
            cat_size=cat_size,
            hidden_layers=hidden_layers,
            seed=self.seed,
            internal_activation_fn=internal_activation_fn,
            output_activation_fn=output_activation_fn,
            batch_size=batch_size,
            layer_norm=layer_norm,
        )

        self.b = DerivedNetwork(
            input_size=input_size,
            output_size=output_size,
            cat_size=cat_size,
            hidden_layers=hidden_layers,
            seed=self.seed,
            internal_activation_fn=internal_activation_fn,
            output_activation_fn=output_activation_fn,
            batch_size=batch_size,
            layer_norm=layer_norm,
        )


    def forward(self, x, action=None):
        """
        Perform a forward propagation inference on environment state vector

        Arguments:
            state - the enviornment state vector

        Returns - action value
        """

        a_out = self.a(x, action)
        b_out = self.b(x, action)

        return a_out, b_out

    def a_net(self, state, action):

        return self.a(state, action)
