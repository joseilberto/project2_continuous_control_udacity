# Project 2 - Continuous Control in the Reacher Environment

<center>
	<img src="https://video.udacity-data.com/topher/2018/June/5b1ea778_reacher/reacher.gif" alt="drawing" width="480"/>
</center>

## Introduction

This project aims to train an armed agent capable of follow an object that moves in the Reacher environment and getting as many rewards as possible. In this environment, the agent only collects a positive reward when it touches the object, otherwise it receives no reward. 

The agent is a double-jointed arm that can translate and rotate in the space. The state vector in each time has 33 elements representing positions, rotations, velocities and angular velocities. In each time step, the agent should take an action that has 4 inputs which apply torques to the joints and change the state of the arm in the next time step.

Here, we present the solution for 2 different versions of the same environment: one with a single arm trying to reach the object and another one with 20 arms each one attempting to reach the nearest object. Please, for a more detailed information about the states, actions and the details of the algorithm used to train the agent refer to the report.pdf file.

## Files in project

|  File | Description | 
|-------|-------------|
| Continous_Control.ipynb  | A notebook where we visualize the environment, train the agent and assess its performance. | 
| agents.py  | A python script where we define the modified DDPG agent. | 
| models.py  | A python script where we define the neural networks (Actor and Critic) used by the agent to estimate the action-value function. |
| report.pdf  | A more complete report of the environment, data and results |
| single_arm_models | Saved weights of the agent in the single arm environment for both actor and critic | 
| multiple_arms models | Saved weights of the agents in the multiple arms environment for all actors and critics | 

## Installation and requirements

Plese, visit the Deep Reinforcement Learning [repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) maintained by Udacity in order to install all dependencies to work with the code used here. All the steps presented there can be break down into the steps in file requirements.txt:

```console
youruse@yourcomputer:~$ conda env create -f environment.yml
youruse@yourcomputer:~$ python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

Finally we should download and extract the reacher environment in the same folder of your project. Please, refer to the list below to download it:

Version with one agent:
- Linux: [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- Mac OSX: [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
- Windows (32-bit): [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
- Windows (64-bit): [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

Versionn with multiple agents:
- Linux: [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- Mac OSX: [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows (32-bit): [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- Windows (64-bit): [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

## The agent

Our agent uses a modified version of the Deep Deterministic Policy Gradients (DDPG) algorithm the paper [Continuous Control with deep reinforcement learning](https://arxiv.org/pdf/1509.02971.pdf) to train the agent on the environment and collect the highest reward during every single step continuously improving it in each episode.

### Training

The agents for the single arm environment and the multiple arm environment are trained in the Continuous_Control.ipynb notebook running the cells presented in its training section. A typical training routine yields a result as the one below in which the agent with the parameters given in the parameters_dict variable in the notebook is capable of reaching an average score of 30 at the end of the last 100 episodes. 

For a single agent:
![The rewards obtained by a single arm at the end of each episode as well as the average of the last 100 episodes and a moving average with step size equal to 3.](https://github.com/joseilberto/project2_continuous_control_udacity/blob/master/images/scores_single_arm.png)

For 20 agents:
![The average rewards obtained by all 20 arms at the end of each episode as well as the average of the last 100 episodes and a moving average with step size equal to 3.](https://github.com/joseilberto/project2_continuous_control_udacity/blob/master/images/scores_multiple_arms.png)

## Report file

The report.pdf file is a more detailed document in which the detailed information of the agent and the neural networks used are presented. We also discuss further the results found and what improvements can be performed. 




