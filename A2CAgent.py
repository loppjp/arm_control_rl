from math import exp
import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from fc_network import Network
from experience import Experience, ExperienceBuffer, ExperienceNotReady

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class A2CModel:

    def __init__(self, state_size, action_size, training_params):

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(training_params["SEED"])
        self.histories = training_params["HISTORIES"]
        self.bootstrap = training_params["BOOTSTRAP"]
        self.training_params = training_params
        self.state_history_queue = deque(maxlen=self.histories)

        randgen = lambda: 2 * (np.random.random_sample((1, self.state_size)) - 0.5)

        [self.state_history_queue.append(randgen()) for x in range(0, self.histories)]

        self.mode = self.training_params["MODE"]

        # Policy Network
        self.policy_net = Network(
            self.state_size * self.histories,
            self.action_size,
            seed=self.training_params["SEED"],
            batch_size=self.training_params["BATCH_SIZE"],
        ).to(device)

        # State Value Network
        self.state_value_net = Network(
            self.state_size * self.histories,
            1,
            cat_size=self.action_size,
            seed=self.training_params["SEED"],
            batch_size=self.training_params["BATCH_SIZE"],
            output_activation_fn=F.leaky_relu,
        ).to(device)

        # Policy Network - for training
        self.policy_net_target = Network(
            self.state_size * self.histories,
            self.action_size,
            seed=self.training_params["SEED"],
            batch_size=self.training_params["BATCH_SIZE"],
        ).to(device)

        # State Value Network - for training
        self.state_value_net_target = Network(
            self.state_size * self.histories,
            1,
            cat_size=self.action_size,
            seed=self.training_params["SEED"],
            batch_size=self.training_params["BATCH_SIZE"],
            output_activation_fn=F.leaky_relu,
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

        self.state_history_queue.append(state)

        squeezed = np.vstack(np.array(list(self.state_history_queue)))

        # adjust for batch size on axis 1
        state_tensor = torch.from_numpy(np.expand_dims(squeezed, 0)).float().flatten(1).to(device)

        self.policy_net.eval()

        with torch.no_grad():

            probs = self.policy_net.forward(state_tensor)

        self.policy_net.train()

        return probs.cpu().data.numpy()


    def update_state_value_estimate(self, state, action, next_state, reward, dones):
        """
        Update the value functio neural network by computing
        the action function at the next state, given the reward
        and gamma
        """

        # compute the state values 

        #next_state_tensor = torch.FloatTensor(next_state).to(device)

        argmax_a_q_sp = self.policy_net_target.forward(next_state)

        max_a_q_sp = self.state_value_net_target.forward(next_state, action=argmax_a_q_sp)

        target_q_sa = reward + self.gamma_tensor * max_a_q_sp * (1 - dones)

        q_sa = self.state_value_net.forward(state, action=action)

        td_error = q_sa - target_q_sa.detach()

        value_loss = td_error.pow(2).mul(0.5).mean()

        self.state_value_optimizer.zero_grad()

        value_loss.backward()

        self.state_value_optimizer.step()

        #V_s_estimated = self.state_value_net.forward(
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
        self.soft_update(self.state_value_net, self.state_value_net_target, self.training_params["TAU"])                     


    def update_policy(self, state, action, next_state, reward) -> None:
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

        #V_s = self.state_value_net.forward(state, action=action)

        #V_s_prime = self.state_value_net.forward(next_state, action=action)

        #A = reward + self.gamma_tensor * (V_s_prime - V_s)
        #A = reward_tensor + self.gamma_tensor * V_s_prime - V_s

        argmax_a_q_s = self.policy_net.forward(state)

        max_a_q_s = self.state_value_net.forward(state, action=argmax_a_q_s)

        # target 0 loss?
        #loss = self.loss(A, torch.zeros_like(A))
        policy_loss = -max_a_q_s.mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step() 

        # ------------------- update target network ------------------- #
        #self.soft_update(self.policy_net, self.policy_net_target, self.training_params["TAU"])                     
        self.soft_update(self.policy_net, self.policy_net_target, self.training_params["TAU"])                     

    def soft_update(self, local_model, target_model, tau):
        """
        Use tau to determine to what extent to update target network

        θ_target = τ*θ_local + (1 - τ)*θ_target

        Arguments:
            local_model - pytorch neural network model. used for actions
            target_model - pytorch neural network model. used for training
            tau - ratio by which to update target from local
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


    def save(self):
        """
        Save model weights to disk
        """

        torch.save(self.policy_net.state_dict(), 'policy_model.pth')
        torch.save(self.state_value_net.state_dict(), 'state_value.pth')


class A2CAgent:

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
        
        # epsilon starts out high, explore
        action = None
        if np.random.random() < epsilon: 
            action = np.random.random((1, self.action_size))

            # undesired, scale the action space from -1 to 1
            # itd be nice to get this from the environment..maybe you can
            action = action * 2 - 1
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
        #do_train = self.t_step % self.model.training_params["UPDATE_TARGET_NET_STEPS"]
        do_train = True
        
        if "TRAIN" == self.model.training_params["MODE"].upper() and do_train:

            try:

                experiences = self.mem_buffer.sample()

                states, actions, rewards, next_states, done = experiences

                self.model.update_state_value_estimate(
                    states,
                    actions,
                    next_states,
                    rewards,
                    done
                )

                self.model.update_policy(
                    states,
                    actions,
                    next_states,
                    rewards
                )

            except ExperienceNotReady:
                pass # not quite ready to train, its ok ...
