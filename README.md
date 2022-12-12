# AI-Pathfinder
# Path Finder 
Using Reinforcement Learning algorithm: A simple Q-learning and Q-Learning with a heuristic

This repository contains code to train an agent which can navigate in a grid to reach its Goal state by chosing an optimal path. Agent is trained using Q-learning algorithm. Performance of the whole training is optimised using a heuristic function.

Below is the grid environment where agent has to navigate to reach the goal state 5.


![Path Finder](https://user-images.githubusercontent.com/26816532/145638233-fc396cb5-3f89-43fa-89a8-3c6f7628016f.png)

## Consider the below scenario to demonstrate the agent's expected learning
Start state: 10
Transisition path avaliable
  - Path#1 10->9->3->1->5
  - Path#2 10->9->3->2->6->7->5
  - Path#3 10->9->8->4->3->1->5
  
Here Path#1 is the optimal path, post complete training of the agent, it should choose the path#1 to reach the goal ignoring the other paths.

# Why Q-learning:
  - A model-free algorithm is an algorithm that estimates the optimal policy without using or estimating the dynamics (transition and reward functions) of the environment.

## Limitations of simple Q-learing algorithm:
  - It takes more time to train the agent for larger grids in real world scenarios.
  - This would increase the time and computation complexities.

## Heuristic function used in this project:
 - To overcome the limitations of simple Q-learning model, we have developed a heuristic function which is able to train the agent quickly and more efficiently.
 - Using this function, agent will not revisit the states and explores a new a path to reach the goal everytime until it finishes the environment's exploration. This way agent's training is faster and q-table is updated in less number of episodes.

## Evaluation metrics:
  - This project includes Learning rate metrics to show agent's learning in simple Q-learing as well as Q-learning with heuristic function.
  - Total Rewards for both the algorithms are also presented.
  - Normalised Q-table with updated rewards at the end of the training.
