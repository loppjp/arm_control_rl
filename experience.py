from collections import deque
from math import exp
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

        self.buffer_len=buffer_length
        self.memory = deque(maxlen=self.buffer_len)
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

       low_idx = self.history_len - 1 
       hi_idx = len(self.memory) - (self.bootstrap_steps - 1)

       valid_indicies = [x for x in range(low_idx, hi_idx)]

       return valid_indicies


    def add_single(self, experience:Experience) -> None:

        self.memory.append(experience)

        self._track_rewarding_indices(experience)


    def _track_rewarding_indices(self, experience:Experience) -> None:

        if len(self.memory) > self.buffer_len:

            self.rewarding_idicies = [x - 1 for x in list(self.rewarding_idicies)]

        if experience.get_reward() > 0.0:

            self.rewarding_idicies.append(len(self.memory) - 1)


    def _valid_sample(self, idx):

        history_inbounds = (idx - (self.history_len - 1)) > -1 and idx < len(self.memory)

        bootstrap_inbounds = idx > -1 and idx + (self.bootstrap_steps - 1) < len(self.memory)

        return history_inbounds and bootstrap_inbounds
               


    def _try_get_valid_sample(self) -> Experience:

        if len(self.memory) < ((self.history_len - 1) + (self.bootstrap_steps - 1) + 1):

            raise ExperienceNotReady("not enough histories")
        
        else:

            return random.choice(self._get_valid_indicies())

            #idx = random.randrange(0, len(self.memory))

            #while not self._valid_sample(idx):

            #    idx = random.randrange(0, len(self.memory))

            #return idx

    def _get_random_experience(self) -> List[Experience]:

        idx = self._try_get_valid_sample()

        self.idx = idx

        experience:List[Experience] = self._get_experices(idx)

        return experience

    
    def _get_reward_weighted_sample(
            self,
        ):

        if random.random() < 0.5 or len(self.rewarding_idicies) < 1:

            return self._get_random_experience()

        else:

            result_set = set(self._get_valid_indicies()) & set(self.rewarding_idicies)

            if not result_set:
                raise ExperienceNotReady("not enough histories to produce reward weighted sample")

            idx = random.choice(list(result_set))

            return self._get_experices(idx)


    def _get_random_sample(
            self, 
            force_reward=True,
            attempts=1) -> List[Experience]:

        _attempts = attempts

        reward_sum = 0

        while force_reward and _attempts > 0:

            experiences = self._get_random_experience()

            reward_sum = sum([x.get_reward() for x in experiences])

            if reward_sum > 0:

                if DEBUG["REWARD_SEARCH"]:
                    print("found reward")

                return experiences

            _attempts = _attempts - 1

        if DEBUG["REWARD_SEARCH"] and force_reward and _attempts <= 0 and reward_sum <= 0:

            print("failed to find usable reward")
            

        # get a random experience.. despite presence of reward signal
        return self._get_random_experience()

    
    def _get_experices(self, idx) -> List[Experience]:

        if self.history_len == 1 and self.bootstrap_steps == 1:
            return [self.memory[idx]]
        elif self.bootstrap_steps == 1:
            return list(self.memory)[
                idx - (self.history_len + 1) : idx + 1
            ]
        else:
            return list(self.memory)[
                idx - (self.history_len - 1) : idx + (self.bootstrap_steps)
            ]


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

        experiences:List[Experience] = self._get_reward_weighted_sample()

        assert len(experiences) > 0, "need non 0 number of samples"

        # history_len x state_space
        # action_space
        # 1
        # state_space
        # 1

        states = np.vstack([x.get_state() for x in experiences[0:self.history_len]]).flatten()

        action = np.array(experiences[self.history_len-1].get_action())

        #debug
        debug_rewards = [x.get_reward() for x in experiences]

        if len(experiences) == 1:
            next_states = np.vstack([x.get_next_state() for x in experiences]).flatten()
        else:
            next_states = np.vstack([x.get_next_state() for x in experiences[self.history_len:]]).flatten()

        # inspired by the n-step bootstrapping code here:
        # https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/component/replay.py
        ecils = slice( self.history_len - 1, None, 1 )
        reward = 0
        done = 1
        #for _reward, _done in zip([x.get_reward() for x in experiences[ecils]], [x.get_done() for x in experiences[ecils]]):
        for _reward, _done in zip([x.get_reward() for x in experiences[ecils]], [x.get_done() for x in experiences[ecils]]):
            reward = reward + self.gamma * _reward * (1 - int(_done))
            done = _done and done

        if DEBUG["LEARNING_REWARD"]:
            with np.printoptions(
                formatter={'float': '{:+2.6f}'.format}
                ): print(f" learning reward: {reward} ")

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
