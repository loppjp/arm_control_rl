from typing import List, Callable

import torch
from torch  import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Network(nn.Module):

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
        internal_activation_fn:Callable=nn.ReLU,
        #internal_activation_fn=None,
        output_activation_fn:Callable=nn.Tanh,
        #output_activation_fn=None,
        batch_size:int = 1,
        layer_norm:bool=False,
        batch_norm:bool=False
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
        self.seed = self.seed_func(seed)
        self.input_size = input_size
        self.output_size = output_size
        self.cat_size = cat_size
        self.batch_size = batch_size
        self.hidden_layers = hidden_layers
        self.batch_norm = batch_norm

        self.batch_norm_layers = []
        self.network_layers = []

        self.internal_activation_fn = internal_activation_fn
        self.output_activation_fn = output_activation_fn

        self.layer_norm = layer_norm

        """
        Deep network with batch normalization 
        between fully connected layers
        """

        self.network_layers.append(nn.Linear(input_size + cat_size, self.hidden_layers[0]))

        self.network_layers.append(self.internal_activation_fn())

        for layer_idx in range(0, len(self.hidden_layers)-1):

            self.network_layers.append(nn.Linear(self.hidden_layers[layer_idx], self.hidden_layers[layer_idx+1]))

            self.network_layers.append(self.internal_activation_fn())


        self.network_layers.append(nn.Linear(self.hidden_layers[-1], self.output_size))

        if self.output_activation_fn:

            self.network_layers.append(self.output_activation_fn())

        self.network_layers = nn.Sequential(*self.network_layers)



    def forward(self, x, action=None):
        """
        Perform a forward propagation inference on environment state vector

        Arguments:
            state - the enviornment state vector

        Returns - action value
        """

        # if actions were supplied.. 
        if action is not None:

            # concatonate the actions on the 1st deminsion
            x = torch.cat((x, action), dim=1)

        x = self.network_layers(x)

        return x

    def seed_func(self, seed):

        torch.manual_seed(seed)