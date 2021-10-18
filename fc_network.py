from numpy.core.defchararray import add
import torch
from torch  import nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Network(nn.Module):

    def __init__(self, 
        input_size:int, 
        output_size:int, 
        seed:int,
        add_sigmoid: bool = False,
        add_tanh: bool = False,
        batch_size: float = 1
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
        self.seed = torch.manual_seed(seed)
        self.add_sigmoid = add_sigmoid
        self.batch_size = batch_size
        self.add_tanh = add_tanh

        """
        Deep network with batch normalization 
        between fully connected layers
        """
        
        self.fc1 = nn.Linear(input_size, 128)

        if batch_size > 1:
            self.bn1 = nn.BatchNorm1d(128)


        self.fc2 = nn.Linear(128, 64)

        if batch_size > 1:
            self.bn2 = nn.BatchNorm1d(64)


        self.fc3 = nn.Linear(64, 64)

        if batch_size > 1:
            self.bn3 = nn.BatchNorm1d(64)


        self.fc4 = nn.Linear(64, 32)

        if batch_size > 1:
            self.bn4 = nn.BatchNorm1d(32)


        self.fc5 = nn.Linear(32, output_size)

        self.sigmoid = nn.Sigmoid()

        self.tanh = nn.Tanh()



    def forward(self, state):
        """
        Perform a forward propagation inference on environment state vector

        Arguments:
            state - the enviornment state vector

        Returns - action value
        """

        x = self.fc1(state)
        
        if self.batch_size > 1:
            x = self.bn1(x)

        x = F.leaky_relu(x)


        
        x = self.fc2(x)

        if self.batch_size > 1:
            x = self.bn2(x)

        x = F.leaky_relu(x)


        
        x = self.fc3(x)

        if self.batch_size > 1:
            x = self.bn3(x)

        x = F.leaky_relu(x)


        
        x = self.fc4(x)

        if self.batch_size > 1:
            x = self.bn4(x)

        x = F.leaky_relu(x)
        


        x = self.fc5(x)       

        if self.add_sigmoid:
            x = self.sigmoid(x)

        if self.add_tanh:
            x = self.tanh(x)

        return x
