import argparse
from collections import deque
import datetime
import json
import os
from typing import NamedTuple
from pathlib import Path
import time

import numpy as np

from unityagents import UnityEnvironment

from td3_train_params import *

from TD3Agent import Agent
from TD3Agent import Model

def get_datefmt_str():

    n = datetime.datetime.now()

    return n.strftime('%d%m%Y_%H%M%S')


def reset_and_describe_environment(env: UnityEnvironment, train_mode: bool = True) -> None:
    """
    Given a 'reset' UnityEnvironment, describe the number
    of agents, the action space and state space

    Args:
        env: a unity environment to interact with
        train_mode: mode to operate, gets passed through to env.reset's kwarg. Defaults to training 
                    mode True

    Returns:
        AllBrainInfo: A Data structure corresponding to the initial reset state of the environment
                      for API see: https://github.com/udacity/deep-reinforcement-learning.git 
                           deep-reinforcement-learning/python/unityagents/brain.py
    """

    # reset the environment
    all_brain_info = env.reset(train_mode=train_mode)

    #print(f"Number of Agents: {len(all_brain_info)}")

    return all_brain_info


class AgentData(NamedTuple):
    agent: Agent = None


class AgentEpisodeData(NamedTuple):
    episode_score: float = 0.0


def train_eval(
        env: UnityEnvironment,
        max_timesteps : int              = DEFAULT_MAX_TIMESTEPS,
        num_episodes : int               = DEFAULT_NUM_EPISODES,
        score_window_episodes : int      = DEFAULT_SCORE_WINDOW_EPISODES,
        reward_solution_criteria : float = DEFAULT_REWARD_SOLUTION_CRITERIA,
        epsilon_start : float            = DEFAULT_EPSILON_START,
        epsilon_end : float              = DEFAULT_EPSILON_END,
        epsilon_decay : float            = DEFAULT_EPSILON_DECAY_FACTOR,
        experience_buffer : int          = DEFAULT_EXPERIENCE_BUFFER,
        do_not_save:bool                 = False,
        load_model:bool                  = False, 
        visual:bool                      = False,
) -> dict:

    """
    Train the agent(s) on the environment

    args:
        env - the UnityEnvironment to use for training
        score_window_size - last N scores to average in rolling window
        reward_solution_criteria - The reward value that represents that the 
                                  environment has been solved
        epsilon_start (float): starting value of epsilon, for epsilon-greedy action selection
        epsilon_end (float): minimum value of epsilon
        epsilon_decay (float): multiplicative factor (per episode) for decreasing epsilon


    returns: score metadata object with training details for plotting
    """

    #### scoring and termination ####
    scores = []         # list containing scores from each episode

    scores_window = deque(maxlen=score_window_episodes)

    solved_episode = -1

    # track scores per step
    step_score_window = deque(maxlen=DEFAULT_STEP_SCORE_WINDOW)

    max_score = 0 # the maximum reward acheived

    consecutive = 0 # the number of consecutive episodes for which the reward has
                    # surpassed the required criteria

    #### epsilon value tracking for epsilon-greedy policy ####
    epsilon = epsilon_start

    #### agent initialization ####

    # dictionary of Agents indexed by environment brain name
    agent_dict = {}

    """
    Reset the environment using the env.reset function.
    Make sure the environment knows we're about to train
    Print the description and get all brain metadata back
    in the AllBrainInfo object
    """
    all_brain_info = reset_and_describe_environment(env, train_mode=True)

    # for now, assume models homogeneous
    brain_key = list(all_brain_info.keys())[0]

    state_size = env.brains[brain_key].vector_observation_space_size

    action_size = env.brains[brain_key].vector_action_space_size

    TRAINING_PARAMS["EXPERIENCE_BUFFER"] = experience_buffer

    # only instantiate 1 model for now
    model = Model(
        state_size,
        action_size,
        TRAINING_PARAMS
    )

    if load_model:
        TRAINING_PARAMS["MODE"] = "EVAL"
        model.load()

    # for each brain in the environment: 
    #   store them off so we can access them by name later
    #   Also instantiate an Agent
    for brain_name in all_brain_info:

        # instantiation and storage for the agent
        agent_dict[brain_name] = AgentData(
            agent=Agent(
                model,
            )
        )

    # delete the scores log if it exists
    if os.path.exists('scores.csv'): os.remove('scores.csv')

    # training loop:
    for episode in range(1, num_episodes+1):

        all_brain_info = reset_and_describe_environment(env, train_mode=True)

        score = 0

        agent_episode_data = {
            brain_name: AgentEpisodeData() for brain_name in all_brain_info
        }

        # timesteps
        for t in range(max_timesteps):

            # dictionary of brain_name to action for this timestep
            step_actions = { }

            for brain_name in all_brain_info:

                if not all_brain_info[brain_name].local_done[0]:

                    # store the chosen action for each brain in the step_actions dictionary
                    action = agent_dict[brain_name].agent.act(
                        all_brain_info[brain_name].vector_observations[0],
                        epsilon
                    )

                    step_actions[brain_name] = action

                    assert action.shape[1] == action_size

                # else, no action taken for done agants, seems like None is ok in many cases
                else:
                    print("act done")


            # step the environment for all agents
            if len(step_actions) > 0:

                next_all_brain_info = env.step(step_actions)

                # update step for each agent
                for brain_name in next_all_brain_info:

                    # update agents that were done at the beggining of this timestep
                    if not all_brain_info[brain_name].local_done[0]:

                        reward = next_all_brain_info[brain_name].rewards[0]

                        step_score_window.append(reward)

                        agent_dict[brain_name].agent.step(
                            all_brain_info[brain_name].vector_observations[0], # state
                            step_actions[brain_name], # action
                            next_all_brain_info[brain_name].rewards[0], # reward
                            next_all_brain_info[brain_name].vector_observations[0], # next state
                            next_all_brain_info[brain_name].local_done[0], # done
                        )

                        if visual: time.sleep(0.001)

                    else:
                        print("step done")


                # update agent scores
                for brain_name in next_all_brain_info:

                    # update agents that were done at the beggining of this timestep
                    if not all_brain_info[brain_name].local_done[0]:

                        agent_episode_data[brain_name] = AgentEpisodeData(
                            agent_episode_data[brain_name].episode_score + next_all_brain_info[brain_name].rewards[0]
                        )

                    else:

                        print("done agent episode data")

                # stop this episode if all the brains have reached a local done
                if all([all_brain_info[brain_name].local_done[0] for brain_name in all_brain_info]): break

                # step training state
                all_brain_info = next_all_brain_info
            
            else:

                break # episode is done since we no agents took steps

            print(f"Step {t:3d}/{max_timesteps}", end="")
            print(f"\tEpisode {episode}/{num_episodes}", end="")
            if len(scores_window) > 0:
                print(f"\tAverage Score: {np.mean(scores_window):.2f}", end="")
                print(f"\tMaxScore: {np.max(max_score):.2f}", end="")
            else:
                print(f"\tAverage Score: {0:.2f}", end="")
                print(f"\tMaxScore: {0:.2f}", end="")
            if len(step_score_window) > 100:
                print(f"\tAvg Step Score-100: {np.mean(list(step_score_window)[-100:]):.6f}", end="")
                print(f"\tStd Dev Step Score-100: {np.std(list(step_score_window)[-100:]):.6f}", end="")
            print(f"", end="\r")


        # the minimum score over all agents is used for criteria computation
        score = min( [ agent_episode_data[brain_name].episode_score for brain_name in all_brain_info ] )             
        scores_window.append(score)
        scores.append(score)

        with open('scores.csv', 'a') as f:
            f.write(f"{score:2.6f}\n")

        # the maximum score acheived over agent minumums
        max_score = max(scores_window)

        # update epsilon
        epsilon = max(epsilon_end, epsilon_decay * epsilon)

        # compute average reward over last 100 episodes
        if np.mean(scores_window) >= reward_solution_criteria:
            consecutive += 1
        else:
            consecutive = 0

        # store solved episode, dont "erase" unless consecutive goes back to 0
        if consecutive > 0 and solved_episode < 0:
            solved_episode = episode
        elif solved_episode > 0 and consecutive == 0:
            solved_episode = -1
            
        # add a few more episodes to this, 150 instead of 100
        if consecutive >= score_window_episodes:
            print(f'\nEnvironment solved in {solved_episode} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
            if not do_not_save:
                model.save()
            break
    print("")

    if not do_not_save:
        model.save()

    return scores



def main(
    max_timesteps: int                = DEFAULT_MAX_TIMESTEPS,
    num_episodes: int                 = DEFAULT_NUM_EPISODES,
    score_window_episodes: int        = DEFAULT_SCORE_WINDOW_EPISODES,
    reward_solution_criteria: float   = DEFAULT_REWARD_SOLUTION_CRITERIA,
    epsilon_start: float              = DEFAULT_EPSILON_START,
    epsilon_end: float                = DEFAULT_EPSILON_END,
    epsilon_decay: float              = DEFAULT_EPSILON_DECAY_FACTOR,
    multi_agent: bool                 = DEFAULT_MULTI_AGENT,
    experience_buffer: int            = DEFAULT_EXPERIENCE_BUFFER,
    model_name: str                   = "model",
    do_not_save:bool                  = False,
    visual:bool                       = False,
    load_model:bool                   = False,
) -> None:
    """
    Outerloop for training function. 

    Access the environment from disk
    """

    if multi_agent and not visual:
        ENV_NAME = 'data/Reacher_Linux_NoVis/Reacher.x86_64'
    elif not multi_agent and not visual:
        ENV_NAME = 'data/Reacher_One_Linux_NoVis/Reacher.x86_64'
    elif multi_agent and visual:
        ENV_NAME = 'data/Reacher_Linux/Reacher.x86_64'
    elif not multi_agent and visual:
        ENV_NAME = 'data/Reacher_One_Linux/Reacher.x86_64'

    if "DATA_PATH" in os.environ:

        ENV_NAME = Path(os.environ["DATA_PATH"])/Path(ENV_NAME)
        print(f"setting ENV_NAME: {ENV_NAME}")

    else:

        ENV_NAME = Path("/")/ENV_NAME

    ENV_NAME=str(ENV_NAME)

    print(f"ENV_NAME: {ENV_NAME}")

    env = UnityEnvironment(file_name=ENV_NAME)

    scores = train_eval(
        env,
        max_timesteps=max_timesteps,
        num_episodes=num_episodes,
        score_window_episodes=score_window_episodes,
        reward_solution_criteria=reward_solution_criteria,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        experience_buffer=experience_buffer,
        do_not_save=do_not_save,
        load_model=load_model,
        visual=visual,
    )

    if not do_not_save:

        score_meta = {}
        score_meta["max"] = np.max(scores)
        score_meta["min"] = np.min(scores)
        score_meta["episodes"] = len(scores)
        score_meta["mean"] = np.mean(scores)
        score_meta["scores"] = scores

        json_data = json.dumps(score_meta)

        # save evaulation results
        with open(f"{model_name}_{get_datefmt_str()}.json", 'w') as f:
            f.write(json_data)

        import matplotlib
        import matplotlib.pyplot as plt

        # plot the scores
        matplotlib.use('Agg')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.savefig(f'{model_name}_{get_datefmt_str()}.png')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        "Train an RL agent on the reacher environment"
    )

    parser.add_argument(
        '-t',
        '--max-timesteps',
        help= 'the maximum number of time steps per episode for which to train ' + \
             f'defaults to {DEFAULT_MAX_TIMESTEPS}',
        default=DEFAULT_MAX_TIMESTEPS,
        type=int
    )

    parser.add_argument(
        '-n',
        '--num-episodes',
        help= 'the number of episodes for which to train ' + \
             f'defaults to {DEFAULT_NUM_EPISODES}',
        default=DEFAULT_NUM_EPISODES,
        type=int
    )

    parser.add_argument(
        '-s',
        '--score-window-episodes',
        help= 'The number of consecutive episodes that must satisfy the reward ' + \
              'solution criteria to consider the task as solved ' + \
             f'defaults to {DEFAULT_SCORE_WINDOW_EPISODES}',
        default=DEFAULT_SCORE_WINDOW_EPISODES,
        type=int
    )

    parser.add_argument(
        '-r',
        '--reward-solution-criteria',
        help= 'the minimum reward value that must be attained in consecutive episodes ' + \
              'until the consecutive count reaches the value prescribed by score window episodes '
             f'defaults to {DEFAULT_REWARD_SOLUTION_CRITERIA}',
        default=DEFAULT_REWARD_SOLUTION_CRITERIA,
        type=float
    )

    parser.add_argument(
        '-e',
        '--epsilon-start',
        help= 'the initial value of the epsilon; the probability to take a random exploratory action ' + \
              'must be between 0 and 1 ' + \
             f'defaults to {DEFAULT_EPSILON_START}',
        default=DEFAULT_EPSILON_START,
        type=float
    )

    parser.add_argument(
        '-ee',
        '--epsilon-end',
        help= 'the minimum value for which epsilon is allowed to decay ' + \
              'must be between 0 and 1 ' + \
             f'defaults to {DEFAULT_EPSILON_END}',
        default=DEFAULT_EPSILON_END,
        type=float
    )

    parser.add_argument(
        '-d',
        '--epsilon-decay',
        help= 'the factor by which to multiply epsilon after an episode for annealing  ' + \
             f'defaults to {DEFAULT_EPSILON_DECAY_FACTOR}',
        default=DEFAULT_EPSILON_DECAY_FACTOR,
        type=float
    )

    parser.add_argument(
        '-b',
        '--experience-buffer',
        help= 'The length of the experience replay buffer ' + \
             f'defaults to {DEFAULT_EXPERIENCE_BUFFER}',
        default=DEFAULT_EXPERIENCE_BUFFER,
        type=int
    )

    parser.add_argument(
        '-m',
        '--multi-agent',
        help= 'Optional parameter to enable multi-agent training ' + \
             f'defaults to {DEFAULT_MULTI_AGENT}',
        default=DEFAULT_MULTI_AGENT,
        type=bool
    )

    parser.add_argument(
        '-x',
        '--do-not-save',
        help='If set, model artifacts, such as the model weights and images will not be saved',
        action='store_true',
        default=False
    )

    parser.add_argument(
        '-v',
        '--visual',
        help='Instantiate graphical environment to watch agent',
        action='store_true',
        default=False,
    )

    parser.add_argument(
        '-l',
        '--load_model',
        help='Load the model from disk',
        action='store_true',
        default=False
    )

    args = parser.parse_args()

    main(
        max_timesteps=args.max_timesteps,
        num_episodes=args.num_episodes,
        score_window_episodes=args.score_window_episodes,
        reward_solution_criteria=args.reward_solution_criteria,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        experience_buffer=args.experience_buffer,
        multi_agent=args.multi_agent,
        do_not_save=args.do_not_save,
        visual=args.visual,
        load_model=args.load_model,
    )
