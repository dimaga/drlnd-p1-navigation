# Project 1: Navigation
### Learning Algorithm

#### Background

Reinforcement Learning studies how to program agents so that they learn by interacting with their environment: play a computer game or control a self driving car to avoid dynamic obstacles in a dense traffic.

Agents observe their world, perform actions and occasionally receive rewards. However, they do not know which sequences of actions lead to rewards. In this project, an agent lives in the banana world. It learns how to collect yellow bananas, but avoid blue bananas without any pre-programming, all by itself.

A book _Reinforcement Learning An Introduciton, Second Edition, by Richard S. Sutton and Andrew G. Barto_ provides a great introduction to reinforcement learning algorithms.

In this project, reinforcement learning is implemented via a Q-function. Q-function predicts what a _cumulative_ reward the agent may obtain if the agent performs in state S an action A.

![formula](https://render.githubusercontent.com/render/math?math=Q(S,A))

_Cumulative_ reward is a sum of rewards that the agent receives when performing a sequence of actions till the end of the episode. Collecting bananas in the envrionment of this project is _episodic_. It means, that after a number of steps the of the agent with the environment interaction ends, and the agents respawns again in the new environment.

#### Deep Q Learning

#### Rainbow

TODO: The report clearly describes the learning algorithm, along with the chosen hyperparameters. It also describes the model architectures for any neural networks.

### Plot of Rewards

TODO: A plot of rewards per episode is included to illustrate that the agent is able to receive an average reward (over 100 episodes) of at least +13. The submission reports the number of episodes needed to solve the environment.

### Ideas for Future Work

TODO: The submission has concrete future ideas for improving the agent's performance.
