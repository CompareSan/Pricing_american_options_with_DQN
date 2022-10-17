from reply_buffer import ReplyBuffer
import numpy as np
from neural_network import build_dqn


class Agent:
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size, input_dims,
                 epsilon_dec=0.99, epsilon_end=0.01, mem_size=1000000, fname='dqn_model.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_name = fname

        self.memory = ReplyBuffer(mem_size, input_dims, n_actions, discrete=True)

        self.q_eval = build_dqn(alpha, n_actions, input_dims, 4, 4)
        self.q_network = build_dqn(alpha, n_actions, input_dims, 4, 4)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        rand = np.random.random()

        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(state, verbose=0)
            action = np.argmax(actions)

        return action

    def learn(self, counter):

        if self.memory.mem_count < self.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        action = np.argmax(action, axis=1)

        q_eval = self.q_eval.predict(state, verbose=0)
        q_next = self.q_network.predict(new_state, verbose=0)

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size)

        q_target[batch_index, action] = reward + self.gamma * np.max(q_next, axis=1) * done
        _ = self.q_eval.fit(state, q_target, verbose=0)

        if counter % 1000 == 0:
            self.q_network.set_weights(self.q_eval.get_weights())

        self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

    def save_model(self):
        # self.q_eval.save(self.model_name)
        pass

    def load_model(self):
        # load_model(self.model_name, compile=False)
        pass
