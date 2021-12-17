from collections import deque
from numbers import Number
import random

from typing import NamedTuple, Tuple, List

import numpy as np
import torch
from torch._C import _TensorBase
from torch.functional import Tensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

        self.memory = deque(maxlen=buffer_length)
        self.seed = random.seed(seed)
        self.state_conversion = state_conversion
        self.action_conversion = action_conversion
        self.gamma = gamma

        # how many times to sample
        self.batch_size = batch_size

        self.history_len = history_len

        self.bootstrap_steps = bootstrap_steps

        self.prebatch = []

    def add_single(self, experience:Experience) -> None:

        self.memory.append(experience)

    def _valid_sample(self, memory, idx):

        return (idx - self.history_len) > 0 and (idx + self.bootstrap_steps) > 1

    def _try_get_valid_sample(self) -> Experience:

        if len(self.memory) < self.history_len + self.bootstrap_steps:

            raise ExperienceNotReady("not enough histories")
        
        else:

            idx = random.randrange(0, len(self.memory))

            while not self._valid_sample(self.memory, idx):

                idx = random.randrange(0, len(self.memory))

            return idx

    def _get_sample_batch(self) -> ExperienceSample:

        batch = [self._get_single_sample() for _ in range(0, self.batch_size)]

        # batch x history_len x state_space
        # batch x action_space
        # batch x 1
        # batch x state_space
        # batch x 1

        states = np.array([b.get_states() for b in batch])
        actions = np.array([b.get_actions() for b in batch])
        rewards = np.array([b.get_rewards() for b in batch])
        next_states = np.array([b.get_states() for b in batch])
        dones = np.array([b.get_dones() for b in batch])

        return ExperienceSample(
            states,
            actions,
            rewards,
            next_states,
            dones
        )

    def _get_single_sample(self) -> ExperienceSample:

        idx = self._try_get_valid_sample()

        experiences:List[Experience] = list(self.memory)[
            idx - (self.history_len + 1) : idx + (self.bootstrap_steps)
        ]

        # history_len x state_space
        # action_space
        # 1
        # state_space
        # 1

        states = np.vstack([x.get_state() for x in experiences[0:self.history_len]]).flatten()

        action = np.array(experiences[-1].get_action())

        next_states = np.vstack([x.get_next_state() for x in experiences[(-1-self.history_len):-1]]).flatten()

        # inspired by the n-step bootstrapping code here:
        # https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/component/replay.py
        ecils = slice( None, -1 , self.history_len - 1 )
        reward = 0
        done = 1
        for _reward, _done in zip([x.get_reward() for x in experiences[ecils]], [x.get_done() for x in experiences[ecils]]):
            reward = _reward + _done * self.gamma * reward
            done = _done and done

        return ExperienceSample(
            states,
            action, 
            reward, 
            next_states, 
            done
        )

    def sample(self) -> Tuple[Tensor]:

        experience_batch:List[ExperienceSample] = self._get_sample_batch()

        sampled = (
            torch.from_numpy(
                experience_batch.get_states()
            ).float().to(device),
            torch.from_numpy(
                experience_batch.get_actions()
            ).float().to(device),
            torch.from_numpy(
                experience_batch.get_rewards()
            ).float().to(device),
            torch.from_numpy(
                experience_batch.get_next_states()
            ).float().to(device),
            torch.from_numpy(
                experience_batch.get_dones()
            ).int().to(device)
        )

        return sampled

    def _base_conv_func(
            conv_func, 
            accessor, 
            experiences:List[Experience]
    ) -> Tensor:

        #dd = [accessor(data) for data in experiences]

        #print(f"accessor: {accessor}")
        #print(dd)

        converted = []

        for experience in experiences:

            converted.append(conv_func(
                torch.from_numpy(
                    np.vstack([
                        accessor(data) for data in experience
                    ])
                )
            ).to(device))

        # [batch_size, sample size, ]
        torch_converted = torch.squeeze(torch.stack(converted))

        return torch_converted

    def __len__(self):

        return len(self.memory)
