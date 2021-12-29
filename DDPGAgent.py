from math import exp
from pathlib import Path
import random
from collections import deque
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from fc_network import Network
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

class DDPGModel:

    def __init__(self, state_size, action_size, training_params):

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(training_params["SEED"])
        self.histories = training_params["HISTORIES"]
        self.bootstrap = training_params["BOOTSTRAP"]
        self.training_params = training_params
        self.state_history_queue = deque(maxlen=self.histories)

        randgen = lambda: 2 * (np.random.random_sample((1, self.state_size)) - 0.5)

        [self.state_history_queue.append(randgen()) for _ in range(0, self.histories)]

        self.mode = self.training_params["MODE"]

        # Policy Network
        self.policy_net_online = Network(
            self.state_size * self.histories,
            self.action_size,
            seed=self.training_params["SEED"],
            batch_size=self.training_params["BATCH_SIZE"],
        ).to(device)

        # soft update only, no training
        self.policy_net_online.eval()

        # Policy Network - for training
        self.policy_net_train = Network(
            self.state_size * self.histories,
            self.action_size,
            seed=self.training_params["SEED"],
            batch_size=self.training_params["BATCH_SIZE"],
        ).to(device)

        # State Value Network
        self.state_value_net_online = Network(
            self.state_size * self.histories,
            1,
            cat_size=self.action_size,
            seed=self.training_params["SEED"],
            batch_size=self.training_params["BATCH_SIZE"],
            #output_activation_fn=F.leaky_relu,
            output_activation_fn=lambda x:x,
        ).to(device)

        # soft update only, no training
        self.state_value_net_online.eval()

        # State Value Network - for training
        self.state_value_net_train = Network(
            self.state_size * self.histories,
            1,
            cat_size=self.action_size,
            seed=self.training_params["SEED"],
            batch_size=self.training_params["BATCH_SIZE"],
            #output_activation_fn=F.leaky_relu,
            output_activation_fn=lambda x:x,
        ).to(device)

        self.policy_optimizer = torch.optim.Adam(
            self.policy_net_train.parameters(), 
            lr=self.training_params["LEARNING_RATE"]
        )

        self.state_value_optimizer = torch.optim.Adam(
            self.state_value_net_train.parameters(), 
            lr=self.training_params["LEARNING_RATE"]
        )

        self.gamma_tensor = torch.FloatTensor([self.training_params["GAMMA"]]).to(device)


    def sample_action(self, state: np.array):
        """
        Sample the policy network by accessing its final output layer
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
            next_state, 
            reward, 
            dones,
        ):
        """
        Update the value functio neural network by computing
        the action function at the next state, given the reward
        and gamma
        """

        # compute the state values 

        #next_state_tensor = torch.FloatTensor(next_state).to(device)

        #argmax_a_q_sp = self.policy_net_target(next_state)

        self.policy_net_online.eval()
        #self.policy_net_target.eval()

        argmax_a_q_sp = self.policy_net_online(next_state)

        max_a_q_sp = self.state_value_net_online(next_state, action=argmax_a_q_sp)

        # 1-dones is already accounted for
        target_q_sa = reward + max_a_q_sp * (1 - dones)

        # reward adjusted in experience recall
        #target_q_sa = reward

        q_sa = self.state_value_net_train(state, action=action)

        td_error = q_sa - target_q_sa.detach()

        value_loss = td_error.pow(2).mul(0.5).mean()

        if DEBUG["VALUE_LOSS"]:
            with np.printoptions(
                formatter={'float': '{:+2.6f}'.format}
                ): print(f" value_loss: {value_loss.cpu().data.numpy()} ")

        self.state_value_optimizer.zero_grad()

        value_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.state_value_net_train.parameters(), 
                                       float('inf'))

        self.state_value_optimizer.step()

        #V_s_estimated = self.state_value_net(
        #    next_state,
        #    action=action
        #)

        #reward_tensor = torch.FloatTensor(reward).to(device)

        #V_s_prime_actual = reward + self.gamma_tensor * V_s_estimated



        #loss = self.loss(V_s_prime_actual, V_s_estimated)
        #self.state_value_optimizer.zero_grad()
        #loss.backward()
        #self.state_value_optimizer.step()

        # ------------------- update target network ------------------- #
        #self.soft_update(self.state_value_net, self.state_value_net_target, self.training_params["TAU"])                     
        self.soft_update(
            self.state_value_net_online, 
            self.state_value_net_train, 
            self.training_params["TAU"]
        )



    def update_policy(
            self, 
            state, 
            action, 
            next_state, 
            reward,
        ) -> None:
        """
        Update the policy neural network by computing the 
        state value function at s (state before env step)
        and the next state (state after env step) given
        the reward and gamma
        """

        #state_tensor = torch.FloatTensor([state]).to(device)
        
        #next_state_tensor = torch.FloatTensor([next_state]).to(device)

        #reward_tensor = torch.FloatTensor([reward]).to(device)

        #print(f"update_policy: state.size(): {state.size()}")

        #V_s = self.state_value_net(state, action=action)

        #V_s_prime = self.state_value_net(next_state, action=action)

        #A = reward + self.gamma_tensor * (V_s_prime - V_s)
        #A = reward_tensor + self.gamma_tensor * V_s_prime - V_s

        #argmax_a_q_s = self.policy_net_target(state)

        #max_a_q_s = self.state_value_net_target(state, action=argmax_a_q_s)

        #argmax_a_q_s = self.policy_net_target(state)
        argmax_a_q_s = self.policy_net_train(state)

        max_a_q_s = self.state_value_net_online(state, action=argmax_a_q_s)

        # target 0 loss?
        #loss = self.loss(A, torch.zeros_like(A))
        policy_loss = -max_a_q_s.mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net_train.parameters(), 
                                       float('inf'))  
        self.policy_optimizer.step() 

        # ------------------- update target network ------------------- #
        #self.soft_update(self.policy_net, self.policy_net_target, self.training_params["TAU"])                     
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
        with open(policy_net, 'r') as f:
            self.policy_net_online = torch.load(f)
            self.policy_net_train = torch.load(f)

        with open(state_value_net, 'r') as f:
            self.state_value_net_online = torch.load(f)
            self.state_value_net_train = torch.load(f)


def sum_data(tensor):

    z = [x.cpu().detach().numpy().sum() for x in tensor]

    if isinstance(z, Iterable):

        z = np.sum(z)

    return float(z)

class DDPGAgent:

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

            #print(" rand ", end="")

            noise = (np.random.normal(
                loc=0,
                scale=self.model.training_params["POLICY_NOISE"],
                size=(1, self.action_size)
            ))

            action += noise

            action = np.clip(action, 0 , 1.0)

        # if output activation sigmoid, adjust
        action = action * 2 - 1

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

        self.mem_buffer.add_single(
            Experience(
                state,
                action,
                reward,
                next_state,
                done
            )
        )

        self.t_step = self.t_step + 1
        do_train = self.t_step % self.model.training_params["TRAIN_INTERVAL_STEPS"] == 0
        do_train = do_train and (self.t_step > self.model.training_params["LEARN_START"])
        #do_train = True
        


        if "TRAIN" == self.model.training_params["MODE"].upper() and do_train:

            for _ in range(0, self.model.training_params["TRAIN_PASSES"]):

                try:

                    experiences = self.mem_buffer.sample()

                    states, actions, rewards, next_states, done = experiences

                    if DEBUG["SAMPLED_REWARD"]:
                        with np.printoptions(
                            formatter={'float': '{:+2.6f}'.format}
                            ): print(f" reward: {rewards.cpu().data.numpy()} ")

                    self.model.update_state_value_estimate(
                        states,
                        actions,
                        next_states,
                        rewards,
                        done,
                    )

                    self.model.update_policy(
                        states,
                        actions,
                        next_states,
                        rewards,
                    )

                except ExperienceNotReady:
                    pass # not quite ready to train, its ok ...

                if DEBUG["NETWORKS"]:

                    self.last_value_net_sum = self.model.policy_net_online.parameters()

                    policy_net_train_sum = sum_data(self.model.policy_net_train.parameters())
                    policy_net_online_sum = sum_data(self.model.policy_net_online.parameters())
                    state_value_net_train_sum = sum_data(self.model.state_value_net_train.parameters())
                    state_value_net_online_sum = sum_data(self.model.state_value_net_online.parameters())

                    print(f"policy_net_train_sum: {policy_net_train_sum:4.6f} ", end="")
                    print(f" policy_net_online_sum: {policy_net_online_sum:4.6f}", end="")
                    print(f" state_value_net_train_sum: {state_value_net_train_sum:4.6f}", end="")
                    print(f" state_value_net_online_sum: {state_value_net_online_sum:4.6f}")