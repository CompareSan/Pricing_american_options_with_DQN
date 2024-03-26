from typing import Tuple

import numpy as np


class ReplyBuffer:
    def __init__(
        self,
        max_size: int,
        batch_size: int,
        state_shape: int,
        n_actions: int,
        discrete: bool = False,
    ) -> None:
        self.mem_size = max_size
        self.mem_count = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, state_shape))
        self.new_state_memory = np.zeros((self.mem_size, state_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.batch_size = batch_size

    def store_transition(self, state, action, reward, new_state, done) -> None:
        index = self.mem_count % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions  # Convert to One-hot encoding
        else:
            self.action_memory[index] = action  # Already in One-hot encoding.

        self.mem_count += 1

    def sample_buffer(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        max_mem = min(self.mem_count, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size)

        states = self.state_memory[batch]
        action = self.action_memory[batch]
        reward = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, action, reward, states_, terminal
