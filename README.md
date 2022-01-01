# Ball/Target Reaching Reinforcement Learning Agent

This repository contains a pytorch based implementation of a policy gradient
agent that can be trained to solve a ball/target reaching task in an [ml-agents](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md) like
environment.

</br>

![Agent GIF](./arm_control.gif)

</br>

</br>

## Project Details:

</br>
In this environment, the agent is a simulated robot arm tasked with following a spherical ball around in a circle. In a given episode the ball can be traveling at a different speed.
</br>

### Agent State Space:
</br>
The reaching agent must provide a set of 4 real valued actions to the environment
</br>
0: torque joint 1 direction 1 </br>
1: torque joint 1 direction 2 </br>
2: torque joint 2 direction 1</br>
3: torque joint 2 direction 2</br>
</br>

### Agent Observaton Space:
</br>
The reacher agent is able to observe an a state vector from the environment composed of 33 elements.
The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm
</br>
</br>
### Agent Reward Structure and Solution
</br>
The agent receives a reward of +0.1 for its "hand" joint being in the proper location for a given timestep
</br>
</br>

## Project Dependencies:
</br>

* Ideally, GPU hardware and access to NVIDIA CUDA
    *  This can be facilitated by setting up an nvidia-docker container based on nvidia/cuda 

### Steps:
</br>

1) Setup a python virtual environment. For example:

    ```
    python -m venv env
    source env/bin/activate
    ```

2) Clone dependencies and install into docker environment:

    </br>

   * Open AI GYM

   </br>

    ```
    git clone https://github.com/openai/gym.git
    cd gym
    pip install -e .
    ```
    </br>

   * Udacity Deep Reinforcement Learning

   </br>

    ```
    git clone https://github.com/udacity/deep-reinforcement-learning.git
    cd deep-reinforcement-learning/python
    pip install .
    ```

3) Download the standalone Unity based training environment for your use case:

    * [Linux, with visuals](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Reacher/one_agent/Reacher_Linux.zip)
    * [Linux, headless, for training](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Reacher/one_agent/Reacher_Linux_NoVis.zip)

    Be sure to unzip each. There are others available on that server for Windows

    Note, the software in this repository expects these Unity environments to be unzipped to a mounted directory called data. E.g.:
    ```
    /data/Reacher_Linux
    /data/Reacher_Linux_NoVis
    ```

## Agent Training:

This section describes the steps to training the agent.

The agent can be trained from the command line or a jupyter-notebook.

* After sourcing the python environment the jupyter-notebook can be started from the command line by calling 
```
jupyter-notebook --ip <IP of host> --port <Port of host>
```
* Alternatively training can be executed from the command line by running:
```
python -m train
```

### Results
* After train is run, a file called scores.csv is written to disk with information about how the training proceeded
* The policy and state value models are saved to .pth file with their respective names. Examples are in the repo
* You can use plot.ipynb to plot the graph using the scores.csv file