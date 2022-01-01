from collections import deque
from numbers import Number
import random

from typing import NamedTuple, Tuple, List

import numpy as np
import torch
from torch._C import _TensorBase
from torch import Tensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DEBUG_REWARD_SEARCH=False
DEBUG_LEARNING_REWARD=False

DEBUG= {
    "REWARD_SEARCH":False,
    "LEARNING_REWARD":False
}

class Experience(NamedTuple):
    state:Tuple[float]=None
    action:Tuple[float]=None
    reward:float=None
    next_state:Tuple[float]=None
    done:Number=None

    def get_state(self): return self.state
    def get_action(self): return self.action
    def get_reward(self): return self.reward
    def get_next_state(self): return self.next_state
    def get_done(self): return self.done


class ExperienceSample(NamedTuple):
    states:np.array # batch x history_len x state_space
    actions:np.array # batch x action_space
    rewards:np.array # batch x 1
    next_states:np.array # batch x state_space
    dones:np.array # batch x 1

    def get_states(self): return self.states
    def get_actions(self): return self.actions
    def get_rewards(self): return self.rewards
    def get_next_states(self): return self.next_states
    def get_dones(self): return self.dones


class ExperienceNotReady(Exception):
    pass

class ExperienceBuffer:

    def __init__(
        self,
        state_size:int,
        action_size:int,
        batch_size:int,
        buffer_length:int,
        history_len:int=1,
        bootstrap_steps:int=1,
        gamma:float=0.996,
        state_conversion=_TensorBase.float,
        action_conversion=_TensorBase.float,
        seed:int=1234
    ):
        """
        A buffer of experience tuples

        Implemented with fixed size numpy array and an indexing system that makes 
        it act like a fifo buffer

        There is an alternate implementation where 50% of the time, the batch
        will be generated with only rewarding samples. See:

            _get_reward_weighted_samples

        Args:
            action_size: the size of the action space, either tuple or int
            buffer_length: length of memory buffer
            batch_size: number of examples from buffer to train on
            history_len: number of previous states to retreive when sampling. Defaults
                         to 1 frame of history.
            bootstrap_steps: number of steps to accumulate reward over. Also corresponds
                             to the reported next state. Defaults to 1 step (TD-estimate)
            gamma: discount factor
            state_conversion: function used to create training memory from states
            action_conversion: function used to create training memory from actions
            seed: for random generator
        """

        self.buffer_len=buffer_length

        self.states      = np.zeros((self.buffer_len, state_size))
        self.actions     = np.zeros((self.buffer_len, action_size))
        self.rewards     = np.zeros((self.buffer_len, 1))
        self.next_states = np.zeros((self.buffer_len, state_size))
        self.dones       = np.zeros((self.buffer_len, 1))

        # insertion idx for fifo support
        self.insert_idx = 0

        self.rewarding_idicies= deque(maxlen=1024)        
        self.seed = random.seed(seed)
        self.state_conversion = state_conversion
        self.action_conversion = action_conversion
        self.gamma = gamma

        # how many times to sample
        self.batch_size = batch_size

        self.history_len = history_len

        self.bootstrap_steps = bootstrap_steps

        self.prebatch = []


        #debug
        self.idx = 0


    def _get_valid_indicies(self):

       if self.history_len == 1:
           low_idx = 0
       else:
           low_idx = self.history_len

       hi_idx = (self.insert_idx + 1) - (self.bootstrap_steps - 1)

       valid_indicies = [x for x in range(low_idx, hi_idx)]

       return valid_indicies


    def add_single(self, experience:Experience) -> None:
        """
        Add an experience sample.

        A state, action, reward, next state, done named tuple
        """

        self.states[self.insert_idx, :] = experience.get_state()
        self.actions[self.insert_idx, :] = experience.get_action()
        self.rewards[self.insert_idx] = experience.get_reward()
        self.next_states[self.insert_idx, :] = experience.get_next_state()
        self.dones[self.insert_idx] = experience.get_done()

        self._track_rewarding_indices(experience)

        # act like a fifo
        if self.insert_idx < self.buffer_len:
            self.insert_idx += 1
        else:
            self.states[:-2, :] = self.states[1:-1, :]
            self.actions[:-2, :] = self.actions[1:-1, :]
            self.rewards[:-2] = self.rewards[1:-1]
            self.next_states[:-2, :] = self.states[1:-1, :]
            self.dones[:-2] = self.states[1:-1]


    def _track_rewarding_indices(self, experience:Experience) -> None:
        """
        Track the indicies that have a positive reward and add
        it to a deque of rewarding indicies. Adjust the indicies once
        the fifo fills up
        """

        if self.insert_idx >= self.buffer_len - 1:

            self.rewarding_idicies = [x - 1 for x in list(self.rewarding_idicies)]

        if experience.get_reward() > 0.0:

            self.rewarding_idicies.append(self.insert_idx)


    def _try_get_valid_sample(self, batch_size=1) -> Experience:
        """
        Sample valid indicies given the settings for bootstrapping and
        histories
        """

        if (self.insert_idx + 1) < ((self.history_len - 1) + (self.bootstrap_steps - 1) + 1):

            raise ExperienceNotReady("not enough histories")
        
        else:

            return random.sample(self._get_valid_indicies(), k=batch_size)

    def _get_random_experience(self) -> List[Experience]:
        """
        Get the index corresponding to bootstrapping and history requirements.

        Access the buffer and grab a list of experience tuples
        """

        idx = self._try_get_valid_sample()[0]

        self.idx = idx

        experience:List[Experience] = self._get_experices(idx)

        return experience


    def _get_random_experiences(self, batch_size) -> List[int]:
        """
        return a list of random experience indicies
        """

        return self._try_get_valid_sample(batch_size=batch_size)

    
    def _get_reward_weighted_samples(self, batch_size=1) -> List[Experience]:
        """
        Get a list of experiences based on indicies with a 50% chance
        of random experiences or experiences that only have rewards in them
        """

        if random.random() < 0.5 or len(self.rewarding_idicies) < batch_size:

            indicies = self._get_random_experiences(batch_size=batch_size)

        else:

            result_set = set(self._get_valid_indicies()) & set(self.rewarding_idicies)

            if not result_set:
                raise ExperienceNotReady("not enough histories to produce reward weighted sample")

            indicies = random.sample(list(result_set), k=batch_size)

        return self._get_experience_arrays(indicies)

    def _get_experience_arrays(self, indicies) -> List[Tuple[np.array]]:
        """
        Populate experiences based on history length and bootstrapping requirements
        """

        if self.history_len == 1:

            states = np.vstack(
                [
                    self.states[_idx, :]
                    for _idx in indicies
                ]
            )

            actions = np.vstack(
                [
                    self.actions[_idx, :]
                    for _idx in indicies
                ]
            )

        else:

            states = np.vstack(
                [
                    self.states[_idx-self.history_len:_idx, :].flatten()
                    for _idx in indicies
                ]
            )

            actions = np.vstack(
                [
                    self.actions[_idx, :]
                    for _idx in indicies
                ]
            )

        if self.bootstrap_steps == 1:

            if self.history_len == 1:

                next_states = np.vstack(
                    [
                        self.next_states[_idx, :]
                        for _idx in indicies
                    ]
                )

            else:

                next_states = np.vstack(
                    [
                        self.next_states[_idx-self.history_len:_idx, :].flatten()
                        for _idx in indicies
                    ]
                )


            dones = np.vstack(
                [
                    self.dones[_idx]
                    for _idx in indicies
                ]
            )

            discounted_rewards = np.vstack(
                [
                    self.gamma * self.rewards[_idx]
                    for _idx in indicies
                ]
            )

        else:

            if self.history_len == 1:
                next_states = np.vstack(
                    [
                        self.next_states[_idx+self.bootstrap_steps-self.history_len:_idx+self.bootstrap_steps, :].flatten()
                        for _idx in indicies
                    ]
                )
            else:
                next_states = np.vstack(
                    [
                        self.next_states[_idx+self.bootstrap_steps, :]
                        for _idx in indicies
                    ]
                )

            continues = np.vstack(
                [
                    1 - self.dones[_idx:_idx+self.bootstrap_steps].flatten()
                    for _idx in indicies
                ]
            )

            dones = np.vstack(
                [
                    self.dones[_idx:_idx+self.bootstrap_steps].flatten()
                    for _idx in indicies
                ]
            )

            gamma_vec = np.vstack(
                [
                    self.gamma ** x for x in range(0, self.bootstrap_steps)
                ]
            ).transpose()

            discounts = gamma_vec * continues

            rewards = np.stack(
                [
                    self.rewards[_idx:_idx+self.bootstrap_steps].flatten()
                    for _idx in indicies
                ]
            )

            discounted_rewards = np.sum(rewards*discounts, axis=1)

        return(
            states,
            actions,
            discounted_rewards,
            next_states,
            dones
        )



    def _get_experices(self, idx) -> List[Experience]:

        if self.history_len == 1 and self.bootstrap_steps == 1:
            #return [self.memory[idx]]
            return [Experience(
                self.states[idx],
                self.actions[idx],
                self.rewards[idx],
                self.next_states[idx],
                self.dones[idx]
            )]
        else:
            return [
                Experience(
                    self.states[_idx], 
                    self.actions[_idx],
                    self.rewards[_idx],
                    self.next_states[_idx],
                    self.dones[_idx]
                )
                for _idx in range( idx - (self.history_len - 1), idx + (self.bootstrap_steps) )
            ]


    def sample(self) -> Tuple[Tensor]:

        #experience_batch = self._get_reward_weighted_samples(self.batch_size)

        indicies = self._get_random_experiences(self.batch_size)
        experience_batch = self._get_experience_arrays(indicies)

        sampled = tuple([
            #torch.from_numpy(np.transpose(x)).float().to(device)
            torch.from_numpy(x).float().to(device)
            for x in experience_batch
        ])

        return sampled
