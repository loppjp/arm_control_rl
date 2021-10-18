import random

import numpy as np
import torch

from fc_network import Network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class A2CModel:

    def __init__(self, state_size, action_size, training_params):

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(training_params["SEED"])
        self.training_params = training_params

        self.mode = self.training_params["MODE"]

        # Policy Network
        self.policy_net = Network(
            self.state_size,
            self.action_size,
            self.training_params["SEED"],
            add_tanh=True,
        ).to(device)

        # State Value Network
        self.state_value_net = Network(
            self.state_size,
            1,
            self.training_params["SEED"],
            add_tanh=True,
        ).to(device)

        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(), 
            lr=self.training_params["LEARNING_RATE"]
        )

        self.state_value_optimizer = torch.optim.Adam(
            self.state_value_net.parameters(), 
            lr=self.training_params["LEARNING_RATE"]
        )

        self.loss = torch.nn.MSELoss()

        self.gamma_tensor = torch.FloatTensor([self.training_params["GAMMA"]]).to(device)


    def sample_action(self, state: np.array):
        """
        Sample the policy network by accessing its final output layer
        """

        state_tensor = torch.from_numpy(state).float().to(device)

        self.policy_net.eval()

        with torch.no_grad():

            probs = self.policy_net(state_tensor)

        self.policy_net.train()

        return probs.cpu().data.numpy()


    def update_state_value_estimate(self, next_state, reward):
        """
        Update the value functio neural network by computing
        the action function at the next state, given the reward
        and gamma
        """

        # compute the state values 

        next_state_tensor = torch.FloatTensor([next_state]).to(device)

        V_s_estimated = self.state_value_net.forward(next_state_tensor)

        reward_tensor = torch.FloatTensor([reward]).to(device)

        V_s_prime_actual = reward_tensor + self.gamma_tensor * V_s_estimated

        loss = self.loss(V_s_prime_actual, V_s_estimated)
        self.state_value_optimizer.zero_grad()
        loss.backward()
        self.state_value_optimizer.step()


    def update_policy(self, state, next_state, reward):
        """
        Update the policy neural network by computing the 
        state value function at s (state before env step)
        and the next state (state after env step) given
        the reward and gamma
        """

        state_tensor = torch.FloatTensor([state]).to(device)
        
        next_state_tensor = torch.FloatTensor([next_state]).to(device)

        reward_tensor = torch.FloatTensor([reward]).to(device)

        V_s = self.state_value_net.forward(state_tensor)

        V_s_prime = self.state_value_net.forward(next_state_tensor)

        A = reward_tensor + self.gamma_tensor * V_s_prime - V_s

        # target 0 loss?
        loss = self.loss(A, torch.zeros_like(A))
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step() 


    def save(self):
        """
        Save model weights to disk
        """

        torch.save(self.policy_net.state_dict(), 'policy_model.pth')
        torch.save(self.state_value_net.state_dict(), 'state_value.pth')


class A2CAgent:

    def __init__(self, state_size, action_size, model):
        """
        Construct the agent

        Arguments:
            state_size: An integer to provide the size of the observation
                        state vector
            action_size: An integer to provide the size of the action vector
            training_params: a dictionary of parameters for training
        """

        self.state_size = state_size
        self.action_size = action_size

        print(f"model action_size: {self.action_size}")

        self.model = model

        self.reset()

        # initialize to random
        self.last_action = np.random.random(self.action_size)

    def reset(self):
        """
        Allow agent to reset at the end of the episode
        """

        self.t_step = 0

    def act(self, state, epsilon):
        """
        Given the state of the environment and an epsilon value,
        return the action that the agent chooses
        """

        # epsilon starts out high, explore
        action = None
        if np.random.random() < epsilon: 
            action = np.random.random(self.action_size)            
        else:
            action = self.model.sample_action(state)

        self.last_action = action

        return action

    def step(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        """
        Advance agent timestep and update both the policy and 
        value estimate networks
        """
        self.model.update_state_value_estimate(next_state, reward)

        self.model.update_policy(state, next_state, reward)

