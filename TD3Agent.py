from math import exp
from pathlib import Path
import random
from collections import deque
from typing import Iterable

import numpy as np
import torch

from twin_fc_networks import Twin
from fc_network import Network
from noise import NormalDecayNoise

from experience import Experience, ExperienceBuffer, ExperienceNotReady

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DEBUG = {
    "NETWORKS":False,
    "ACTION":False,
    "STATE":False,
    "VALUE":False,
    "VALUE_LOSS":False,
    "REWARD":False,
    "SAMPLED_REWARD":False
}

class Model:
    """
    Wraps TD3 neural networks. There is an online model used for inference 
    (sometimes) and a training model used for training updates.

    Tested with continuous action spaces.

    TD3 has a policy neural network to generate actions from an observation
    of the environment state. It also has a state value network to estimate the 
    value of a given state and action.

    The state value network is a parallel twin of networks
    """

    def __init__(self, state_size, action_size, training_params):

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(training_params["SEED"])
        self.histories = training_params["HISTORIES"]
        self.bootstrap = training_params["BOOTSTRAP"]
        self.training_params = training_params
        self.state_history_queue = deque(maxlen=self.histories)

        # prepopulate history buffer with random data since we wont have real experiences yet
        randgen = lambda: 2 * (np.random.random_sample((1, self.state_size)) - 0.5)
        [self.state_history_queue.append(randgen()) for _ in range(0, self.histories)]

        self.mode = self.training_params["MODE"]

        # Policy Network
        self.policy_net_online = Network(
            self.state_size * self.histories,
            self.action_size,
            seed=self.training_params["SEED"],
            batch_size=self.training_params["BATCH_SIZE"],
            output_activation_fn=torch.nn.Tanh,
        ).to(device)

        # soft update only, no training
        self.policy_net_online.eval()

        # Policy Network - for training
        self.policy_net_train = Network(
            self.state_size * self.histories,
            self.action_size,
            seed=self.training_params["SEED"],
            batch_size=self.training_params["BATCH_SIZE"],
            output_activation_fn=torch.nn.Tanh,
        ).to(device)

        # State Value Network
        self.state_value_net_online = Twin(
            self.state_size * self.histories,
            1,
            #self.action_size,
            cat_size=self.action_size,
            seed=self.training_params["SEED"],
            batch_size=self.training_params["BATCH_SIZE"],
            output_activation_fn=None,
        ).to(device)

        # soft update only, no training
        self.state_value_net_online.eval()

        # State Value Network - for training
        self.state_value_net_train = Twin(
            self.state_size * self.histories,
            1,
            #self.action_size,
            cat_size=self.action_size,
            seed=self.training_params["SEED"],
            batch_size=self.training_params["BATCH_SIZE"],
            output_activation_fn=None,
        ).to(device)

        # optimizer for policy network
        self.policy_optimizer = torch.optim.Adam(
            self.policy_net_train.parameters(), 
            lr=self.training_params["LEARNING_RATE"]
        )

        # optimizer for state value network
        self.state_value_optimizer = torch.optim.Adam(
            self.state_value_net_train.parameters(), 
            lr=self.training_params["LEARNING_RATE"]
        )

        # ensure policy network training and online network weights match
        self.soft_update(
            self.policy_net_online,
            self.policy_net_train, 
            1.0
        )

        # ensure state value network training and online network weights match
        self.soft_update(
            self.state_value_net_online, 
            self.state_value_net_train, 
            1.0,
        )

        # initialize some tensors to be used later
        self.gamma_tensor = torch.FloatTensor([self.training_params["GAMMA"]]).to(device)
        self.torch_zeros = torch.zeros(()).to(device)
        self.torch_ones = torch.ones(()).to(device)
        self.torch_neg_ones = torch.Tensor([-1.0]).to(device)
        self.policy_noise_clip = torch.FloatTensor([self.training_params["POLICY_NOISE_CLIP"]]).to(device)


    def sample_action(self, state: np.array):
        """
        Sample the policy network to determine the next action

        Args: state - observation space
        """

        self.state_history_queue.append(state)

        squeezed = np.vstack(np.array(list(self.state_history_queue)))

        # adjust for batch size on axis 1
        state_tensor = torch.from_numpy(np.expand_dims(squeezed, 0)).float().flatten(1).to(device)

        with torch.no_grad():

            probs = self.policy_net_online(state_tensor)

        return probs.cpu().data.numpy()


    def update_state_value_estimate(
            self, 
            state, 
            action,
            reward, 
            next_state, 
            dones,
        ):
        """
        Update the value function neural network by computing
        the action function at the next state, given the reward
        and gamma

        Implementation derived from TD3 implementation from Miguel Morales' 
        Grokking Deep Reinforcement Learning. See: 
        https://github.com/mimoralea/gdrl/blob/master/notebooks/chapter_12/chapter-12.ipynb
        """

        with torch.no_grad():

            #a_ran = 1.0 - (-1.0)
            a_ran = self.torch_ones - self.torch_neg_ones

            # target policy smoothing regularization noise.
            # See TD3 Paper section 5.3
            a_noise = torch.rand_like(action) * \
                self.policy_noise_clip * a_ran
            a_noise = a_noise.to(device)

            n_min = self.policy_noise_clip * self.torch_neg_ones

            n_max = self.policy_noise_clip * self.torch_ones

            # clip noise
            a_noise = torch.max(torch.min(a_noise, n_max), n_min)

            argmax_a_q_sp = self.policy_net_online(next_state)

            noisy_argmax_a_q_sp = argmax_a_q_sp + a_noise

            noisy_argmax_a_q_sp = torch.max(
                torch.min(
                    noisy_argmax_a_q_sp, self.torch_ones
                ),
                self.torch_neg_ones
            )

            max_a_q_sp_a, max_a_q_sp_b, *rest = self.state_value_net_online(
                next_state, action=noisy_argmax_a_q_sp
            )

            max_a_q_sp = torch.min(max_a_q_sp_a, max_a_q_sp_b)

            if self.bootstrap > 1:
                target_q_sa = reward + max_a_q_sp
            else:
                target_q_sa = reward + self.gamma_tensor * max_a_q_sp * (1 - dones)


        q_sp_a, q_sp_b, *_ = self.state_value_net_train(state, action)

        td_error_a = q_sp_a - target_q_sa
        td_error_b = q_sp_b - target_q_sa

        value_loss = td_error_a.pow(2).mul(0.5).mean() + td_error_b.pow(2).mul(0.5).mean()

        self.state_value_optimizer.zero_grad()
        value_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.state_value_net_train.parameters(), 
                                       float('inf'))

        self.state_value_optimizer.step()


        if DEBUG["VALUE_LOSS"]:
            with np.printoptions(
                formatter={'float': '{:+2.6f}'.format}
                ): print(f" value_loss: {value_loss.cpu().data.numpy()} ")


    def soft_update_value(self):
        """
        gentle updates to the state value network
        """

        self.soft_update(
            self.state_value_net_online, 
            self.state_value_net_train, 
            self.training_params["TAU"]
        )



    def update_policy(
            self, 
            state, 
            action, 
            reward,
            next_state, 
        ) -> None:
        """
        Update the policy neural network by computing the 
        state value function at s (state before env step)
        and the next state (state after env step) given
        the reward and gamma
        """

        # produces different results
        #argmax_a_q_s = self.policy_net_target(state)
        argmax_a_q_s = self.policy_net_train(state)

        max_a_q_s = self.state_value_net_train.a_net(state, action=argmax_a_q_s)

        policy_loss = -max_a_q_s.mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net_train.parameters(), 
                                       float('inf'))  
        self.policy_optimizer.step() 


    def soft_update_policy(self):
        """
        gentle updates to the state value network
        """

        self.soft_update(
            self.policy_net_online,
            self.policy_net_train, 
            self.training_params["TAU"]
        )

    def soft_update(self, online_model, train_model, tau):
        """
        Use tau to determine to what extent to update train network

        θ_target = τ*θ_local + (1 - τ)*θ_target

        Arguments:
            online_model - pytorch neural network model. used for actions
            train_model - pytorch neural network model. used for training
            tau - ratio by which to update target from local
        """
        for train_param, online_param in zip(train_model.parameters(), online_model.parameters()):
            online_param.data.copy_(tau*train_param.data + (1.0-tau)*online_param.data)


    def save(
            self,
            policy_net_name:Path='policy_model.pth',
            state_value_net_name:Path='state_value.pth',
        ):
        """
        Save model weights to disk
        """

        torch.save(self.policy_net_online.state_dict(), policy_net_name)
        torch.save(self.state_value_net_online.state_dict(), state_value_net_name)

    def load(
            self,
            policy_net:Path='policy_model.pth',
            state_value_net:Path='state_value.pth',
        ):
        """
        Load model weights from disk
        """
        self.policy_net_online.load_state_dict(torch.load(policy_net))
        self.policy_net_train.load_state_dict(torch.load(policy_net))

        self.state_value_net_online.load_state_dict(torch.load(state_value_net))
        self.state_value_net_train.load_state_dict(torch.load(state_value_net))


def sum_data(tensor):

    z = [x.cpu().detach().numpy().sum() for x in tensor]

    if isinstance(z, Iterable):

        z = np.sum(z)

    return float(z)

class Agent:

    def __init__(
            self, 
            model,
    ):
        """
        Construct the agent

        Arguments:
            state_size: An integer to provide the size of the observation
                        state vector
            action_size: An integer to provide the size of the action vector
            training_params: a dictionary of parameters for training
        """

        self.state_size = model.state_size
        self.action_size = model.action_size

        self.model = model


        self.accumlation_state = None
        self.accumlation_action = None
        self.accumlated_rewards = []

        self.reset()

        # memory buffer for experience
        self.mem_buffer = ExperienceBuffer(
            self.state_size,
            self.action_size,
            self.model.training_params["BATCH_SIZE"],
            self.model.training_params["EXPERIENCE_BUFFER"],
            self.model.training_params["HISTORIES"],
            self.model.training_params["BOOTSTRAP"],
            self.model.training_params["GAMMA"],
        )

        # initialize to random
        self.last_action = np.random.random(self.action_size)

        # initialize time step for training updates
        self.t_step = 0

        # Exploration Noise
        self.normal_decay_noise = NormalDecayNoise(
            self.action_size,
            max_noise=self.model.training_params["DECAY_START"],
            min_noise=self.model.training_params["DECAY_END"],
            decay_steps=self.model.training_params["DECAY_STEPS"],
        )

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

        action = self.model.sample_action(state)
        
        if np.random.random() < epsilon: 

            noise = self.normal_decay_noise(self.t_step)

            action += noise

            action = np.clip(action, -1.0, 1.0)

        self.last_action = action

        if DEBUG["STATE"]:
            with np.printoptions(
                formatter={'float': '{:+2.6f}'.format}
                ): print(f" state: {state} ")

        #if not random:
        if DEBUG["ACTION"]:
            with np.printoptions(
                formatter={'float': '{:+2.6f}'.format}
                ): print(f" action: {action} ")

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

        if DEBUG["REWARD"]:
            with np.printoptions(
                formatter={'float': '{:+2.6f}'.format}
                ): print(f" reward: {reward} ")

        with open('sar.csv', 'a') as f:
            [f.write(f"{str(float(x))}, ") for x in state]
            [f.write(f"{str(float(x))}, ") for x in action]
            f.write(f"{reward:+2.6}, ")
            f.write('\n')

        experiece_to_store = Experience(
            state,
            action,
            reward,
            next_state,
            done
        )

        self.mem_buffer.add_single(experiece_to_store)

        #self.replay_buffer.store(experiece_to_store)

        self.t_step = self.t_step + 1

        do_train_value = self.t_step % self.model.training_params["TRAIN_VALUE_STEPS"] == 0
        do_train_value = do_train_value and (self.t_step > self.model.training_params["LEARN_START"])

        do_train_policy = self.t_step % self.model.training_params["TRAIN_POLICY_STEPS"] == 0
        do_train_policy = do_train_policy and (self.t_step > self.model.training_params["LEARN_START"])
        #do_train = True
        


        if "TRAIN" == self.model.training_params["MODE"].upper() and do_train_value:

            for _ in range(0, self.model.training_params["TRAIN_PASSES"]):

                try:

                    experiences = self.mem_buffer.sample()

                    exp_states, exp_actions, exp_rewards, exp_next_states, exp_dones = experiences

                    if DEBUG["SAMPLED_REWARD"]:
                        with np.printoptions(
                            formatter={'float': '{:+2.6f}'.format}
                            ): print(f" reward: {exp_rewards.cpu().data.numpy()} ")

                    self.model.update_state_value_estimate(
                        exp_states,
                        exp_actions,
                        exp_rewards,
                        exp_next_states,
                        exp_dones,
                    )

                except ExperienceNotReady:
                    pass # not quite ready to train, its ok ...


        if self.t_step % self.model.training_params["SOFT_UPDATE_VALUE_STEPS"] == 0:

            self.model.soft_update_value()


        if "TRAIN" == self.model.training_params["MODE"].upper() and do_train_policy:

            for _ in range(0, self.model.training_params["TRAIN_PASSES"]):

                try:

                    experiences = self.mem_buffer.sample()

                    exp_states, exp_actions, exp_rewards, exp_next_states, exp_dones = experiences

                    if DEBUG["SAMPLED_REWARD"]:
                        with np.printoptions(
                            formatter={'float': '{:+2.6f}'.format}
                            ): print(f" reward: {exp_rewards.cpu().data.numpy()} ")

                    self.model.update_policy(
                        exp_states,
                        exp_actions,
                        exp_rewards,
                        exp_next_states,
                    )

                except ExperienceNotReady:
                    pass # not quite ready to train, its ok ...


        if self.t_step % self.model.training_params["SOFT_UPDATE_POLICY_STEPS"] == 0:

            self.model.soft_update_policy()


        if (do_train_policy or do_train_value) and DEBUG["NETWORKS"]:

            self.last_value_net_sum = self.model.policy_net_online.parameters()

            policy_net_train_sum = sum_data(self.model.policy_net_train.parameters())
            policy_net_online_sum = sum_data(self.model.policy_net_online.parameters())
            state_value_net_train_sum = sum_data(self.model.state_value_net_train.parameters())
            state_value_net_online_sum = sum_data(self.model.state_value_net_online.parameters())

            print(f"policy_net_train_sum: {policy_net_train_sum:4.6f} ", end="")
            print(f" policy_net_online_sum: {policy_net_online_sum:4.6f}", end="")
            print(f" state_value_net_train_sum: {state_value_net_train_sum:4.6f}", end="")
            print(f" state_value_net_online_sum: {state_value_net_online_sum:4.6f}")