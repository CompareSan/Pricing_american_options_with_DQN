import copy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from option_pricer.neural_network import DQN
from option_pricer.option_environment import OptionEnvironment
from option_pricer.reply_buffer import ReplyBuffer


class DQNAgent:
    def __init__(
        self,
        env: gym.Env | OptionEnvironment,
        replay_buffer: ReplyBuffer,
        q_network: DQN,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        optimizer: torch.optim,
        device: torch.device,
        gamma: float = 0.99,
        model_name=None,
    ) -> None:

        self.env = env
        self.gamma = gamma
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.action_space = [i for i in range(self.env.action_space.n)]

        self.model_name = model_name
        self.memory = replay_buffer
        self.device = device
        self.q_network = q_network.to(device)
        self.q_target = copy.deepcopy(q_network).to(device)
        self.optimizer = optimizer

    def store(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        new_state: np.ndarray,
        done: np.ndarray,
    ) -> None:
        self.memory.store_transition(state, action, reward, new_state, done)

    def take_action(self, state: np.ndarray) -> int:

        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                actions = self.q_network(state)

            action = torch.argmax(actions).cpu().item()

        return action

    def train(self, counter: int) -> None:

        if self.memory.mem_count < self.memory.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer()
        action = np.argmax(action, axis=1)  # Decode the action from one-hot

        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        new_state = torch.tensor(new_state, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.float32).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).to(self.device)

        q_network_values = self.q_network(state).gather(1, action.long().unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_target_values = self.q_target(new_state).max(dim=1)[0]
        targets = reward + self.gamma * q_target_values * done

        criterion = nn.MSELoss()
        loss = criterion(q_network_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if counter % 500 == 0:
            self.q_target.load_state_dict(self.q_network.state_dict())

        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def save_model(self, path: str) -> None:
        torch.save(self.q_network.state_dict(), path)
