import copy
from typing import List, Protocol

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from option_pricer.neural_network import Policy
from option_pricer.option_environment import OptionEnvironment
from option_pricer.reply_buffer import ReplyBuffer


class Agent(Protocol):
    def train(
        self,
        n_episodes: int,
    ) -> None: ...

    def save_model(
        self,
        path: str,
    ) -> None: ...


class DQNAgent:
    def __init__(
        self,
        env: gym.Env,
        policy_network: Policy,  # q network
        optimizer: torch.optim,
        device: torch.device,
        gamma: float = 0.99,
        initial_epsilon: float = 1,
        final_epsilon: float = 0,
    ) -> None:
        self.env = env
        self.gamma = gamma
        self.epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = (initial_epsilon - final_epsilon) / 25000
        self.action_space = [i for i in range(self.env.action_space.n)]

        self.memory = ReplyBuffer(
        state_shape=self.env.observation_space.shape[0],
        n_actions=self.env.action_space.n,
        )

        self.device = device
        self.policy_network = policy_network.to(device)
        self.q_target = copy.deepcopy(policy_network).to(device)
        self.optimizer = optimizer

    def _store(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        new_state: np.ndarray,
        done: np.ndarray,
    ) -> None:
        self.memory._store_transition(state, action, reward, new_state, done)

    def _take_action(self, state: np.ndarray) -> float:
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                actions = self.policy_network(state)

            action = torch.argmax(actions).cpu().item()

        return action

    def _take_step(self, counter: int) -> None:
        if self.memory.mem_count < self.memory.batch_size:
            return
        state, action, reward, new_state, done = self.memory._sample_buffer()
        action = np.argmax(action, axis=1)  # Decode the action from one-hot

        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        new_state = torch.tensor(new_state, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.float32).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).to(self.device)

        q_network_values = (
            self.policy_network(state).gather(1, action.long().unsqueeze(1)).squeeze(1)
        )
        with torch.no_grad():
            q_target_values = self.q_target(new_state).max(dim=1)[0]
        targets = reward + self.gamma * q_target_values * done

        criterion = nn.MSELoss()
        loss = criterion(q_network_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if counter % 500 == 0:
            self.q_target.load_state_dict(self.policy_network.state_dict())

        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def train(self, n_episodes: int):
        scores = []
        counter = 0
        for i in range(n_episodes):
            done = False
            score = 0
            observation, _ = self.env.reset()

            while not done:
                action = self._take_action(observation)
                observation_, reward, terminated, truncated, _ = self.env.step(action)
                score += reward
                done = terminated or truncated
                self._store(observation, action, reward, observation_, done)
                observation = observation_
                self._take_step(counter)
                counter += 1

            scores.append(score)

            print("episode: {}, score: {}".format(i, score))
            print(self.epsilon)

    def save_model(self, path: str) -> None:
        torch.save(self.policy_network.state_dict(), path)

class ReinforceAgent:
    def __init__(
        self,
        env: gym.Env,
        policy_network: Policy,
        optimizer: torch.optim,
        device: torch.device,
        gamma: float = 0.99,
    ) -> None:
        self.env = env
        self.gamma = gamma
        self.action_space = [i for i in range(self.env.action_space.n)]

        self.device = device
        self.policy_network = policy_network.to(device)
        self.optimizer = optimizer

    def _take_action(self, state: np.ndarray) -> float:
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            actions = self.policy_network(state)

        prob_dist_over_actions = F.softmax(actions).cpu()
        cum_prob_dist = prob_dist_over_actions.cumsum(0)
        action = torch.searchsorted(cum_prob_dist, torch.rand(1)).item()

        return action

    def _take_step(
        self, actions: List[float], states: List[float], rewards: List[float]
    ) -> None:
        cum_reward_from_t = []
        log_policy = []
        for t in range(len(rewards) - 1, 0, -1):
            cum_reward_from_t.append(
                sum(
                    [
                        self.gamma ** (k - t) * rewards[k]
                        for k in range(t, len(rewards))
                    ],
                )
            )

            pred_actions_logits = self.policy_network(
                torch.tensor(states[t], dtype=torch.float32).to(self.device)
            )
            prob_dist_over_actions = F.softmax(pred_actions_logits)
            log_policy.append(torch.log(prob_dist_over_actions)[actions[t]])

        eps = np.finfo(np.float32).eps.item()
        cum_reward_from_t = torch.tensor(cum_reward_from_t, dtype=torch.float32).to(
            self.device
        )
        cum_reward_from_t = (cum_reward_from_t - cum_reward_from_t.mean()) / (
            cum_reward_from_t.std() + eps
        )

        policy_loss = torch.zeros(1).to(self.device)
        for log_policy, cum_reward_from_t in zip(log_policy, cum_reward_from_t):
            policy_loss += -log_policy * cum_reward_from_t

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

    def train(self, n_episodes: int):
        scores = []
        for i in range(n_episodes):
            done = False
            cum_reward = 0
            observation, _ = self.env.reset()
            rewards = []
            actions = []
            states = []
            while not done:
                states.append(observation)
                action = self._take_action(observation)
                observation_, reward, terminated, truncated, _ = self.env.step(action)
                cum_reward += reward
                done = terminated or truncated
                observation = observation_

                rewards.append(reward)
                actions.append(action)

            self._take_step(actions, states, rewards)

            scores.append(cum_reward)

            print("episode: {}, score: {}".format(i, cum_reward))

    def save_model(self, path: str) -> None:
        torch.save(self.policy_network.state_dict(), path)
