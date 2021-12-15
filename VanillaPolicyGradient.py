import random
import math

import numpy as np
from numpy.testing._private.utils import requires_memory
import torch

from fc_network import Network
from experience import ExperienceBuffer, Experience

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VanillaPolicyGradientModel:

    def __init__(
        self,
        state_size,
        action_size,
        training_params
    ):

        self.state_size = state_size
        self.action_size = action_size * 2 # fit gaussian
        self.seed = random.seed(training_params["SEED"])
        self.training_params = training_params

        self.mode = self.training_params["MODE"]

        # Policy Network
        self.policy_net = Network(
            input_size=self.state_size,
            output_size=self.action_size,
            seed=self.training_params["SEED"],
            batch_size=self.training_params["BATCH_SIZE"],
        ).to(device)

        self.value_net = Network(
            input_size=self.state_size,
            output_size=1,
            cat_size=self.action_size,
            seed=self.training_params["SEED"],
            batch_size=self.training_params["BATCH_SIZE"],
        ).to(device)

        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(), 
            lr=self.training_params["LEARNING_RATE"]
        )

        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(),
            lr=self.training_params["LEARNING_RATE"]
        )

        self.experiences = ExperienceBuffer(
            self.training_params["BATCH_SIZE"],
            self.training_params["EXPERIENCE_BUFFER"]
        )

        self.gamma = torch.FloatTensor([self.training_params["GAMMA"]]).to(device)

        # convert initial parameters to fixed range of values
        for p in self.policy_net.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -1.0, 1.0))

        # convert initial parameters to fixed range of values
        for p in self.value_net.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -1.0, 1.0))
    
    def sample_action(self, state: np.array):
        """
        Sample the policy network by accessing its final output layer
        """

        state_tensor = torch.from_numpy((np.expand_dims(state, 0))).float().to(device)

        self.policy_net.eval()

        with torch.no_grad():

            probs = self.policy_net.forward(state_tensor)

        self.policy_net.train()

        return probs.detach().cpu().data.numpy()
        #return probs.clone().detach().cpu().data.numpy()


    def update_state_value_estimate(
        self,
        state,
        action,
        next_state,
        reward, 
        done
    ):

        argmax_a_q_sp = self.policy_net.forward(next_state)

        max_a_q_sp = self.value_net.forward(next_state, action=argmax_a_q_sp)

        target_q_sa = reward + max_a_q_sp * (1 - done)

        q_sa = self.value_net.forward(state, action=action)

        td_error = q_sa - target_q_sa.detach()

        value_loss = td_error.pow(2).mul(0.5).mean()

        self.value_optimizer.zero_grad()

        value_loss.backward()

        self.value_optimizer.step()


    def update(
            self,
            state,
            action,
            reward,
            next_state,
            done,
            verbose=False,
    ):

        self.experiences.add_single(
            Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done
            )
        )

        if len(self.experiences) < self.training_params["BATCH_SIZE"]:
            return 0

        experiences = self.experiences.sample()

        states, actions, rewards, next_states, dones = experiences

        # from DDPG model
        argmax_a_q_sp = self.policy_net(next_states)
        max_a_q_sp = self.value_net(next_states, action=argmax_a_q_sp)
        target_q_sa = rewards + self.gamma * max_a_q_sp * (1 - dones)
        q_sa = self.value_net(states, action=argmax_a_q_sp)
        td_error = q_sa - target_q_sa.detach()
        value_loss = td_error.pow(2).mul(0.5).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()

        self.value_optimizer.step()

        argmax_a_q_s = self.policy_net(states)
        max_a_q_s = self.value_net(states, action=argmax_a_q_s)
        policy_loss = -max_a_q_s.mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()


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


class VanillaPolicyGradientAgent:

    def __init__(
        self,
        model,
        n_step_bootstrap=1
    ):

        self.state_size = model.state_size
        self.action_size = model.action_size

        self.model = model
        self.n_step_bootstrap = n_step_bootstrap

        self.reset()

        # initialize to random
        self.last_action = np.random.random(self.action_size)
        self.explore_exploit = 'explore'

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
        epsilon_sample = np.random.random()

        self.explore_exploit = 'exploit'

        #if False: #np.random.random() < epsilon: 
        if epsilon_sample < epsilon: 
            #print("random")
            params = np.random.random(self.action_size) 

            # undesired, scale the action space from -1 to 1
            # itd be nice to get this from the environment..maybe you can
            #params = params * 2 - 1

            self.explore_exploit = 'explore'
        else:
            #print("det")
            params = self.model.sample_action(state)[0]

        #print(f"params: {params}")

        actions = []

        for idx in range(0, math.floor(len(params)/2)):

            #sigma=(params[idx] + 1) / 2.0
            sigma=params[idx]
            mu=params[idx+1]
            #if idx == 0:
            #    print(f"mu, sigma: {mu} {sigma}")
            actions.append(np.random.normal(mu, sigma))
        actions = np.array(actions)

        actions = np.minimum(np.ones_like(actions), actions)
        actions = np.maximum(np.zeros_like(actions), actions)

        # get gaussian sample to proper scale
        actions = (actions - 0.5) * 2.0

        assert len(actions == math.floor(self.action_size / 2))

        self.last_action = params

        #debug_print = self.print_counter % self.PRINT_INTERVAL
        
        if False:#debug_print == 0:
            print(f"actions: {actions}")
        #print(f"shape: {action.shape}")

        return actions
        #return np.tile(action, (1,4))

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
        #reward = action[0]

        debug_print = 1#self.print_counter % self.PRINT_INTERVAL
        self.print_counter += 1

        #if debug_print == 0:
        if False:
        
            print(f"all time reward: {self.alltime_reward}")

        #self.accumulated_rewards.append(reward)
        #self.accumulated_states.append(state)

        #self.alltime_reward += reward

        #if True: #done or len(self.accumulated_rewards) == self.n_step_bootstrap:
        #if done or len(self.accumulated_rewards) == self.n_step_bootstrap:

        loss = self.model.update(
            state,
            self.last_action,
            reward,
            next_state,
            done,
            verbose = debug_print == 0# deebug
        )

        #action_str = [f"{a:+0.6f}" for a in action]

        #print(f"reward, loss, action, explore_exploit: \
        #    {reward:0.6f} {loss} {action_str} {self.explore_exploit}")