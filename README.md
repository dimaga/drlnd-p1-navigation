# Project 1: Navigation
### Project Details

This is my solution to Navigation Project of Udacity Deep Reinforcement Learning course. Original project template is available at https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation

An agent lives in the world full of yellow and blue bananas. The purpose of the agent is to collect as many yellow bananas as possible while avoiding blue bananas. The agent receives a reward of +1 for collecting a yellow banana, and a reward of -1 for a blue banana.

The environment is considered solved if the agent gets an average cumulative reward of +13 over 100 consecutive episodes while training for less than 1800 episodes.

In the pursue of positive rewards, the agent may perform one of the following discrete actions:
* 0 - move forward
* 1 - move backward
* 2 - turn left
* 3 - turn right

The agent perceives the world as 37-dimensional state vector, which consits of agent's velocity and ray-based distances to objects around forward direction.

To find the best policy, the agent learns Deep Q Network with epsilon-greedy policy and a number of optimization tricks.

The project consists of the following files:

* __Navigation.ipynb__ contains Python code to train and test the agent using different algorithms
* __*.pth__ stores weights of various Deep Q Neural Network configurations
* __Report.md__ provides a description of the implementation

### Getting Started

TODO: The README has instructions for installing dependencies or downloading needed files.

### Instructions

TODO: The README describes how to run the code in the repository, to train the agent.
