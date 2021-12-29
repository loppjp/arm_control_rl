from typing import List, Callable

import torch
from torch  import nn
import torch.nn.functional as F

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
        internal_activation_fn:Callable=F.leaky_relu,
        #internal_activation_fn=F.linear,
        #internal_activation_fn=None,
        #output_activation_fn=torch.tanh,
        output_activation_fn:Callable=torch.sigmoid,
        #output_activation_fn=F.linear,
        #output_activation_fn=F.leaky_relu,
        #output_activation_fn=None,
        batch_size:int = 1,
        layer_norm:bool=True
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
        self.batch_size = batch_size
        self.hidden_layers = hidden_layers

        self.batch_norm_layers = []
        self.network_layers = []

        self.internal_activation_fn = internal_activation_fn
        self.output_activation_fn = output_activation_fn

        self.layer_norm = layer_norm

        """
        Deep network with batch normalization 
        between fully connected layers
        """

        for layer_idx, _ in enumerate(self.hidden_layers):

            # if this is the input layer
            if layer_idx == 0:

                #if self.layer_norm: self.network_layers.append(nn.LayerNorm(input_size + cat_size))

                if batch_size > 1:
                    self.batch_norm_layers.append(nn.BatchNorm1d(input_size + cat_size))

                self.network_layers.append(nn.Linear(input_size + cat_size, self.hidden_layers[0]))

                #if self.layer_norm: self.network_layers.append(nn.LayerNorm(self.hidden_layers[0]))


            # this is an internal layer
            else:

                if batch_size > 1:
                    self.batch_norm_layers.append(nn.BatchNorm1d(self.hidden_layers[layer_idx-1]))

                self.network_layers.append(nn.Linear(self.hidden_layers[layer_idx-1], self.hidden_layers[layer_idx]))

                #if self.layer_norm: self.network_layers.append(nn.LayerNorm(self.hidden_layers[layer_idx]))

        if batch_size > 1:
            self.batch_norm_layers.append(nn.BatchNorm1d(self.hidden_layers[-1]))

        #if layer_idx == len(self.hidden_layers):
        #    self.network_layers.append(nn.Linear(input_size + cat_size, self.hidden_layers[0]))

        #else:
        #self.network_layers.append(nn.Linear(self.hidden_layers[layer_idx-1], self.hidden_layers[layer_idx]))

        self.network_layers.append(nn.Linear(self.hidden_layers[-1], output_size))

        #if batch_size > 1:
        #    self.batch_norm_layers.append(nn.BatchNorm1d(self.hidden_layers[layer_idx]))

        #if self.layer_norm: self.network_layers.append(nn.LayerNorm(output_size))


        self.batch_norm_layers = nn.ModuleList(self.batch_norm_layers)
        self.network_layers = nn.ModuleList(self.network_layers)



    def forward(self, x, action=None):
        """
        Perform a forward propagation inference on environment state vector

        Arguments:
            state - the enviornment state vector

        Returns - action value
        """

        # activate state before concattonation
        if self.internal_activation_fn:
            x = self.internal_activation_fn(x)

        # if actions were supplied.. 
        if action is not None:

            # concatonate the actions on the 1st deminsion
            x = torch.cat((x, action), dim=1)

        # for each layer
        for layer_idx, layer in enumerate(self.network_layers):

            # if batch size is greater than 1
            if self.batch_size > 1:

                # apply batch normalization
                x = self.batch_norm_layers[layer_idx](x)

            # apply the layer update
            x = layer(x)

            # if this is the last layer

            # this is an internal layer
            if self.internal_activation_fn and not isinstance(layer, nn.BatchNorm1d):

                # Applay the internal activation function
                x = self.internal_activation_fn(x)

            if layer_idx == (len(self.network_layers) - 1):

                # apply the output activation function
                x = self.output_activation_fn(x)


        return x
