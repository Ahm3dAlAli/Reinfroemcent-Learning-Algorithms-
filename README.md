# 🤖 Reinforcement Learning: Analysis and Implementation 🎮

Welcome to my reinforcement learning project! This project aims to analyze various reinforcement learning techniques, such as MDP solvers, Monte Carlo, Q-learning, DQN, REINFORCE, and DDPG, and provide insights into their effectiveness and implementation.

## 📋 Table of Contents

<ul>
  <li><a href="#introduction">Introduction</a></li>
  <li><a href="#literature">Literature Review</a></li>
  <li><a href="#data">Data Collection and Preparation</a></li>
  <li><a href="#metho">Methodology</a></li>
  <li><a href="#analysis">Data Analysis</a></li>
  <li><a href="#recommendations">Recommendations</a></li>
  <li><a href="#implementation">Implementation Examples</a></li>
</ul>

<a name="introduction"></a>

## 🚀 Introduction

Reinforcement learning (RL) is an essential branch of artificial intelligence that focuses on training agents to make decisions based on interactions with their environment. The goal is to maximize the cumulative reward over time. Various techniques have been developed to solve RL problems, and this project aims to analyze and provide insights into their effectiveness and implementation.

<a name="literature"></a>

## 📚 Literature Review

As part of this project, a literature review was conducted to find relevant articles related to the reinforcement learning techniques being studied. Here are some of the most relevant ones:

* Sutton and Barto (2018) - Reinforcement Learning: An Introduction.
* Mnih et al. (2015) - Human-level control through deep reinforcement learning.
* Williams (1992) -imple statistical gradient-following algorithms for connectionist reinforcement learning.
* Watkins and Dayan (1992) - Q-learning.
* Kaelbling et al. (1996) - Reinforcement learning: A survey.
* Silver et al. (2014) - Deterministic Policy Gradient Algorithms.
* Schulman et al. (2015) - Trust Region Policy Optimization.
* Lillicrap et al. (2015) - Continuous control with deep reinforcement learning.
<a name="data"></a>

## 🌐 Data Collection and Preparation

Data for reinforcement learning problems often comes in the form of environments provided by libraries like OpenAI Gym, PyBullet, or custom-built environments. These environments offer a standardized interface for agents to interact with and receive feedback in the form of rewards. Before implementing RL algorithms, it is essential to understand the environment's state and action spaces, along with any specific rules or limitations.

<a name="metho"></a>

## 📝 Methodology

To investigate the effectiveness of various reinforcement learning techniques, we will implement and compare the following algorithms:

1) MDP solvers
2) Monte Carlo methods
3) Q-learning
4) Deep Q-Network (DQN)
5) REINFORCE
6) Deep Deterministic Policy Gradient (DDPG)
Each algorithm will be tested in a variety of environments to analyze their performance under different circumstances. The comparison will be based on factors such as learning speed, stability, and final performance.

<a name="analysis"></a>

## 📈 Data Analysis

Various analysis techniques will be used to evaluate the performance of the reinforcement learning algorithms. The analysis includes comparing learning curves, measuring the average cumulative reward, and evaluating the stability of the algorithms. By analyzing these factors, we can better understand the strengths and weaknesses of each method in different scenarios.

In this project, we've found the optimal hyperparameters for the following algorithms:

Monte Carlo with a performance of 7.888 : 
* "epsilon": 0.6
* "gamma": 0.99

Q-Learning with a performance of 60.349 : 
* "epsilon": 0.6
* "gamma": 0.99
* frequency: 5000

REINFORCE (on Bipedal environment) with a performance of -317.601: 
* learning rate: 6e-3 

DQN (on CartPole environment) with performances of 398.894 and 373.2, respectively: 
* Epsilon Expoenetial decay strategy,exploration fraction: 0.01
* Epsilon Linear decay strategy ,exploration decay: 0.001

DDPG (on Bipedal environment): 
* Critic Hidden Layer
* Policy Hidden Layer

After tuning the DDPG (on Bipedal environment): 
* Policy leanring rate, 1e-3
* Critic leanring rate, 1e-2
* Critic Hidden Layer, [400,300]
* Policy Hidden Layer, [400,300]
* Gamma, 0.99
* Tau, 0.1
* Batch Size, 128
* Buffer Capacity, 1e6


It's worth mentioning that the tuning time ranges from 2 minutes to 1 hour and 30 minutes on a CPU. 
However, this process can be accelerated by using the NVIDIA GPU toolkit, 
reducing the tuning time to a range of 10 seconds to 30 minutes.

## 💡 Recommendations

Based on the analysis of the various reinforcement learning algorithms, we recommend using the optimal hyperparameters mentioned above to achieve the best performance in each environment.
By adjusting the parameters for each algorithm, you can improve the stability and performance of the learning process.
Furthermore, using NVIDIA GPU toolkit can significantly reduce the time required for hyperparameter tuning, allowing for faster experimentation and optimization.

<a name="conclusion"></a>

## 🏁 Conclusion

This project showcases the use of various reinforcement learning algorithms, such as Monte Carlo, Q-Learning, REINFORCE, DQN, and DDPG, for solving different tasks. 
By analyzing their performance and tuning the hyperparameters, we can achieve better results in each environment. 
Utilizing the NVIDIA GPU toolkit can significantly speed up the tuning process, enabling faster experimentation and optimization.
By understanding the strengths and weaknesses of each algorithm, we can make informed decisions on which method is best suited for a particular task. As reinforcement learning continues to advance, we can expect even more efficient and robust solutions to complex problems.
