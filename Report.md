# Project 1: Navigation
### Learning Algorithm

#### Background

Reinforcement Learning studies how to program agents to learn by interacting with their environment: play a computer game or control a self driving car to avoid dynamic obstacles in a dense traffic.

Agents observe their world, perform actions and occasionally receive rewards. However, they do not know which sequences of actions lead to rewards. In this project, an agent lives in the banana world. It learns how to collect yellow bananas, but to avoid blue bananas without preprogramming of its behavior.

A book _Reinforcement Learning An Introduciton, Second Edition, by Richard S. Sutton and Andrew G. Barto_ provides a great introduction to reinforcement learning algorithms.

In this project, reinforcement learning is implemented via a Q-function. Q-function predicts what a _cumulative_ reward the agent will obtain if the agent performs in a state S an action A:

![formula](https://render.githubusercontent.com/render/math?math=Q(S,A))

During training, the agent explores the world to learn an approximation of the Q-function. During testing, it exploits the Q-function by choosing the action that corresponds to the maximum Q-value for the given state:

![formula](https://render.githubusercontent.com/render/math?math=argmax_a%20Q(S,A))

_Cumulative_ reward is a sum of rewards that the agent receives when performing a sequence of actions till the end of an episode. Normally, future rewards are _discounted_ with some &lambda; factor, which is between 0 and 1. Let's assume that:

* _t_ is a timestep

* _R<sub>t</sub>_ is an immediate reward at t (in this project, +1 for picking a yellow banana, and -1 for picking a blue banana; the negative reward is a punishment)

* _A<sub>t</sub>_ is an agent action (in this project, it is a direction of motion)

* _S<sub>t</sub>_ is the environment state (in this project, it is ray distances to objects in front of the agent and the agent speed)

then:

![formula](https://render.githubusercontent.com/render/math?math=Q(S_t,A_t)%20%3D%20R_t%20%2B%20\lambda%20R_{t%2B1}%20%2B%20\lambda^2%20R_{t%2B2}%20%2B%20\lambda^3%20R_{t%2B3}%20%2B%20...)

The sum propagates until the end of the episode.

Collecting bananas in the envrionment of this project is _episodic_. It means, that after a number of steps when the agent with the environment interaction ends, the agents respawns again in a new version of the banana world and starts collecting rewards from zero again.

The sum above can be written in a recursive form:

![formula](https://render.githubusercontent.com/render/math?math=Q(S_t,A_t)%20%3D%20R_t%20%2B%20\lambda%20Q(S_{t%2B1},A_{t%2B1}))

One of the approaches to learn Q-function would be to initialize it with some small positive values, and linearly converge its recursive form at each step t while the agent is exploring the environment, taking actions _A<sub>t</sub>_, observing states _S<sub>t</sub>_ and receiving rewards _R<sub>t</sub>_:

![formula](https://render.githubusercontent.com/render/math?math=Q(S_t,A_t)%20\leftarrow%20Q(S_t,A_t)%20%2B%20\alpha%20(R_t%20%2B%20\lambda%20max_a%20Q(S_{t%2B1},a)-%20Q(S_t,A_t)))

&alpha; controls the level of convergence. It is a value between 0.0 and 1.0. And the approach just described is called _Q-learning_.

In order to decide an action while training, the agent may generate a uniform random value between 0 and 1, and if it is less than some &epsilon;, choose a random action; otherwise, choose _argmax<sub>a</sub>Q(S<sub>t</sub>, a)_. The former case is called _exploration_, the letter is _exploitation_. And the overall approach is called an &epsilon;-greedy policy.

If &epsilon; is high, the agent prefers to try new things to explore the environment. If &epsilon; is low, the agent searches for the maximum reward under the current knowledge of its Q-value estimate. In this project, &epsilon; is set 1.0 in the beginning of training and slowly converges to 0.01 by the end. See ```train()``` method in ```Navigation.ipynb``` and its ```eps_start``` and ```eps_end``` arguments.

#### Deep Q Learning

In its naive form, Q-function is represented by an array with S and A to be indices of that array. This approach is normally used in the books to illustrate how reinforcement learning works on some toy examples. However, in practise, S or A are usually continous and multi-dimensional with infinite amount of values.

In this project, A is a discrete value with one of four possible choices: go forward, go backward, turn left or turn right. However, S is a 37-dimensional float vector, holding the speed of the agent and distances to objects in front of it. Therefore, the array form of Q-function will not work here. Instead, Q-function is approximated by an Artificial Network.

https://www.deeplearningbook.org/ is a good introduction to Deep Learning, studying Artificial Neural Networks. Neural network consists of matrix multiplications by weights, stacked in layers, with non-linear transformations between them. In a classical example, a neural network takes pixel values as the input and predicts what type of an object it sees. For example, it can distinguish cats and dogs.

In this project, the neural network predicts the cumulative reward of possible actions for the given state. That is, it represents a Q-function. This approach is called Deep Q Network. In 2013, Deep Mind successfully applied it to learn the agent to play Atari games at a super-human level by using screen pixels as state S and applying joystick commands as actions A. See their seminal paper at http://files.davidqiu.com//research/nature14236.pdf

Instead of applying Q-learning recursive convergence formula, given in the previous example, artificial neural network weights are adjusted using a method, called Stochastic Gradient Descent. &alpha; is replaced with a parameter, called _learningRate_. Stochastic Gradient Descent calculates matrices of partial derivatives of weights with respect to _(R_t + &lambda; max<sub>a</sub> Q(S<sub>t+1</sub>,a) - Q(S<sub>t</sub>,A<sub>t</sub>)))_ error. It then adds this partial derivates multiplied by _learningRate_ to weights to minimize the error whenever the neural network sees _S<sub>t</sub>_ next time to produce cumulative rewards for _A<sub>t</sub>_.

Calculating matrices of partial derivatives for a very big expression is manually infeasible. On the top of that, all of the numeric operations take enormous amount of hardware resources and need parallel computations. Often, they should run on GPU or a specialized hardware. Fortunately, there are libraries that automate partial derivative generation and parallel computations. The most popular are https://www.tensorflow.org/ and https://pytorch.org/

This project uses https://pytorch.org/ version 0.4.0. The neural networks have been trained on CPU and double checked on GPU. Since 37-dimensional state vector is relatively small, CPU resources were enough.

Instead of stochastic gradient descent, my project uses Adam optimizer (see https://arxiv.org/abs/1412.6980). It improves the convergence accuracy and speed of stochastic gradient descent by tuning learning rate dependently on the training results and by applying an number of additional heuristics and methods.


#### Rainbow

TODO: The report clearly describes the learning algorithm, along with the chosen hyperparameters. It also describes the model architectures for any neural networks.

### Plot of Rewards

TODO: A plot of rewards per episode is included to illustrate that the agent is able to receive an average reward (over 100 episodes) of at least +13. The submission reports the number of episodes needed to solve the environment.

### Ideas for Future Work

* Try Asynchronous Methods for Deep Reinforcement Learning, explained in https://arxiv.org/abs/1602.01783 - this change requires an update of Unity ML Agents library to at least v. 0.9, as mentioned in https://blogs.unity3d.com/ru/2019/11/11/training-your-agents-7-times-faster-with-ml-agents/

* Replace ray-based obstacle detection with pixelwise input, based on convolutional neural networks, see https://en.wikipedia.org/wiki/Convolutional_neural_network and https://pytorch.org/docs/master/generated/torch.nn.Conv2d.html#conv2d

* Try recurrent neural network with LSTM ( https://en.wikipedia.org/wiki/Long_short-term_memory ) and Transformers ( https://en.wikipedia.org/wiki/Transformer_(machine_learning_model) ) to better learn about the hidden state of the world (behind the agent) and avoid Buridan's ass problem, when the agent gets frozen between two choices and does not dare which action to pick

* Try replacing discrete actions with continous commands, like linear and angular velocity, to smoothly pass through bananas

* Try training the agents in a real environment to pick objects or explore the real world. For example, implement robot vacuum cleaner AI
