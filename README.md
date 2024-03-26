# Option Pricer

This project implements a Deep Q-Network (DQN) agent to price American Put options using reinforcement learning. It utilizes a custom environment to model the option pricing problem and a neural network for the agent's Q-function approximation.

## Overview

The project consists of the following components:

- `neural_network.py`: Defines the DQN neural network used by the agent.
- `reply_buffer.py`: Implements the replay buffer for storing and sampling transitions.
- `agent.py`: Contains the DQNAgent class responsible for training the agent using the DQN algorithm.
- `option_environment.py`: Defines the OptionEnvironment class, representing the option pricing environment.
- `main.py`: Entry point for running the training process and pricing options.


