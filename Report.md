# Project 1: Navigation
### Learning Algorithm

#### Background

Reinforcement Learning studies how to program agents to learn by interacting with their environment: play a computer game or control a self driving car to avoid dynamic obstacles in a dense traffic.

Agents observe their world, perform actions and occasionally receive rewards. However, they do not know which sequences of actions lead to rewards. In this project, an agent lives in the banana world. It learns how to collect yellow bananas, but avoid blue bananas without any pre-programming, all by itself.

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

Collecting bananas in the envrionment of this project is _episodic_. It means, that after a number of steps the of the agent with the environment interaction ends, the agents respawns again in a new version of the banana world and starts collecting rewards from zero again.

The sum above can be written in a recursive form:

![formula](https://render.githubusercontent.com/render/math?math=Q(S_t,A_t)%20%3D%20R_t%20%2B%20\lambda%20Q(S_{t%2B1},A_{t%2B1}))

One of the approaches to learn Q-function would be to initialize it with some small positive values, and linearly converge its recursive form at each step t while the agent is exploring the environment, taking actions _A<sub>t</sub>_, observing states _S<sub>t</sub>_ and receiving rewards _R<sub>t</sub>_:

![formula](https://render.githubusercontent.com/render/math?math=Q(S_t,A_t)%20\leftarrow%20Q(S_t,A_t)%20%2B%20\alpha%20(R_t%20%2B%20\lambda%20max_a%20Q(S_{t%2B1},a)-%20Q(S_t,A_t)))

#### Deep Q Learning

In its naive form, Q-function is represented by an array with S and A to be indices of that array. This approach is normally used in the books to illustrate how reinforcement learning works on a toy examples. However, in practise, S or A are usually continous and multi-dimensional with infinite amount of values.


#### Rainbow

TODO: The report clearly describes the learning algorithm, along with the chosen hyperparameters. It also describes the model architectures for any neural networks.

### Plot of Rewards

TODO: A plot of rewards per episode is included to illustrate that the agent is able to receive an average reward (over 100 episodes) of at least +13. The submission reports the number of episodes needed to solve the environment.

### Ideas for Future Work

TODO: The submission has concrete future ideas for improving the agent's performance.
