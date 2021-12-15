import random

import numpy as np
from numpy.testing._private.utils import requires_memory
import torch

from fc_network import Network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReinforceModelContinuous:

    def __init__(
        self,
        state_size,
        action_size,
        training_params
    ):

        self.state_size = state_size
        self.action_size = action_size
        self.action_size = 2 # mu and sigma for gaussian for 1 output
        self.seed = random.seed(training_params["SEED"])
        self.training_params = training_params

        self.mode = self.training_params["MODE"]

        # Policy Network
        self.policy_net = Network(
            self.state_size,
            self.action_size,
            #self.action_size,
            seed=self.training_params["SEED"],
        ).to(device)

        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(), 
            lr=self.training_params["LEARNING_RATE"]
        )


    
    def sample_action(self, state: np.array):
        """
        Sample the policy network by accessing its final output layer
        """

        state_tensor = torch.from_numpy((np.expand_dims(state, 0))).float().to(device)

        self.policy_net.eval()

        with torch.no_grad():

            probs = self.policy_net.forward(state_tensor)

            probs = probs.cpu().data.numpy()
            
            # 1st dim batch, 2nd dim (mu, sigma, 3rd dim is same as action space)
            probs = np.reshape(probs, (probs.shape[0], 2, -1))

            mu = probs[:, 0, :]
            
            sigma = np.abs(probs[:, 1, :])

            # sample a gaussian with mean mu and std dev sigma
            action = np.random.normal(mu, sigma, mu.shape)

        self.policy_net.train()

        return action


    def update_state_value_estimate(
        self,
        state,
        action,
        next_state,
        reward, 
        done
    ):

        pass

    def update_policy(
            self,
            states,
            action,
            next_state,
            rewards,
            verbose=False,
    ):

        states = np.array(states)

        if len(states.shape) < 1:
            states = np.expand_dims(states, 0)


        actions_tensor = self.policy_net.forward(torch.tensor(states).float().to(device))

        actions_tensor = torch.reshape(actions_tensor, (states.shape[0], 2, -1))

        T = len(rewards)
        
        # discounts over rewards
        discounts = np.logspace(0, T, num=T, base=self.training_params["GAMMA"], endpoint=False)

        discounted_returns = np.array([np.sum(discounts[:T-t] * rewards[t:]) for t in range(T)])

        #discounted_returns = discounts * returns
        #discounted_returns = np.array([np.sum(rewards[t:]) for t in range(T)])

        if verbose:

            print(f"discounted returns: {discounted_returns}")
            print(f"actions_tensor: {actions_tensor}")

        discounted_returns_tensor = torch.tensor(discounted_returns, requires_grad=True).to(device)
        #discounted_returns_tensor = torch.tensor(discounted_returns).to(device)
        #discounted_returns_tensor = actions_tensor

        policy_loss = -(
            discounted_returns_tensor * torch.log(
                torch.normal(
                    actions_tensor[:, 0],
                    torch.abs(actions_tensor[:, 1])
                )
            )
        )
        
        #policy_loss = torch.exp(-discounted_returns_tensor)
        #policy_loss = torch.exp(discounted_returns_tensor)
        #policy_loss = -(discounted_returns_tensor).mean()

        if verbose:
            print(f"policy_loss: {policy_loss.item()}")

        #print(f"policy_loss: {policy_loss}")

        self.policy_optimizer.zero_grad()

        policy_loss.backward()

        self.policy_optimizer.step()

        if False: #verbose:
        #if verbose:

            params = self.policy_net.parameters()
            param = next(params)
            while param is not None:
                print(f"param: {param}")
                try:
                    param = next(params)
                except StopIteration:
                    break


    def _soft_update(
        self, 
        local_model, 
        target_model, 
        tau
    ):

        pass

    def save(self):
        """
        Save model weights to disk
        """

        torch.save(self.policy_net.state_dict(), 'policy_model.pth')


class ReinforceAgentContinuous:

    def __init__(
        self,
        state_size,
        action_size,
        experience_buffer_size,
        model,
        n_step_bootstrap=1
    ):

        self.state_size = state_size
        self.action_size = action_size

        self.model = model
        self.n_step_bootstrap = n_step_bootstrap

        self.reset()

        # initialize to random
        self.last_action = np.random.random(self.action_size)

        self.accumulated_rewards = []
        self.accumulated_states = []


        # debug
        self.alltime_reward = 0
        self.print_counter = 0
        self.PRINT_INTERVAL = 1000


    def reset(self):

        pass

    def act(self, state, epsilon, verbose=False):
        """
        Given the state of the environment and an epsilon value,
        return the action that the agent chooses
        """
        
        # epsilon starts out high, explore
        action = None
        if False: #np.random.random() < epsilon: 
            action = np.random.random(self.action_size)            

            # undesired, scale the action space from -1 to 1
            # itd be nice to get this from the environment..maybe you can
            action = action * 2 - 1
            action = np.expand_dims(action, axis=0)
        else:
            action = self.model.sample_action(state)

        self.last_action = action

        debug_print = self.print_counter % self.PRINT_INTERVAL
        
        if debug_print == 0:
            print(f"action: {action}")
        #print(f"shape: {action.shape}")

        #return action
        return np.tile(action, (1,4))

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

#        if action[0] > 0.45 and action[0] < 0.55 and \
#           action[1] > 0.45 and action[1] < 0.55 and \
#           action[2] > 0.45 and action[2] < 0.55 and \
#           action[3] > 0.45 and action[3] < 0.55:
#           reward = 1
#        else:
#            reward = 0

        #reward = -action[0] -action[1] - action[2]-action[3]
        #reward = action[0] +action[1] + action[2]+action[3]
        reward = -action[0]

        debug_print = self.print_counter % self.PRINT_INTERVAL
        self.print_counter += 1

        #print(f"reward: {reward}")

        if debug_print == 0:
        
            print(f"all time reward: {self.alltime_reward}")

        self.accumulated_rewards.append(reward)
        self.accumulated_states.append(state)

        self.alltime_reward += reward

        if True: #done or len(self.accumulated_rewards) == self.n_step_bootstrap:

            self.model.update_policy(
                self.accumulated_states,
                None,
                None,
                self.accumulated_rewards, 
                verbose = debug_print == 0# deebug
            )

            self.accumulated_states = []
            self.accumulated_rewards = []