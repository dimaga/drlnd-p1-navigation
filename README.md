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

The agent perceives the world as a 37-dimensional state vector, which consists of the agent's velocity and ray-based distances to objects around forward direction.

To find the best policy, the agent learns Deep Q Network with epsilon-greedy policy and a number of optimization tricks.

Besides README.md, this repository holds of the following files:

* __Navigation.ipynb__ contains Python code to train and test the agent using different algorithms
* __*.pth__ stores weights of various Deep Q Neural Network configurations
* __Report.md__ provides a description of the implementation

### Getting Started

Follow the steps, described in https://github.com/udacity/deep-reinforcement-learning/tree/dc65050c8f47b365560a30a112fb84f762005c6b README.md, Dependencies section, to deploy your development environment for this project.

Basically, you will need:

* Python 3.6
* PyTorch 0.4.0
* Numpy and Matplotlib, compatible with PyTorch
* Unity ML Agents. Udacity Navigation Project requires its own version of this environment, available https://github.com/udacity/deep-reinforcement-learning/tree/dc65050c8f47b365560a30a112fb84f762005c6b/python with references to other libraries

The project has been developed and tested on Mac OS Catalina with a CPU version of PyTorch 0.4.0, and in Udacity Workspace with a CUDA version of PyTorch.

### Instructions

If you installed conda environment as described in https://github.com/udacity/deep-reinforcement-learning/tree/dc65050c8f47b365560a30a112fb84f762005c6b README.md, type

```
conda activate drlnd
```

Then activate Jupyter notebook in your browser (Chrome is recommended):

```
jupyter notebook
```

Then open Navigation.ipynb from this repository, look through it and Run cells if necessary. 

* See guidelines how to use Jupyter notebook at https://jupyter.org/
* Look through __Report.md__ of this repository to learn further details about my solution
