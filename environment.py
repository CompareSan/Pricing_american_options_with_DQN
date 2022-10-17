import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class OptionEnvironment:
    def __init__(self, r, sigma, strike, initial_price, today_date, maturity_date, freq):
        self.r = r
        self.sigma = sigma
        self.strike = strike
        self.initial_price = initial_price
        self.day_count = 365.
        self.time_grid = pd.date_range(start=today_date, end=maturity_date, freq=freq).to_pydatetime()
        start = self.time_grid[0]
        self.fraction_list = [(date - start).days / self.day_count for date in self.time_grid]# Fractions of years for discount

        self.state = self.initial_price, self.fraction_list[0]
        self.history_states = [self.state]

    def step(self, action, ind):
        price, t = self.state
        if self.isterminal(t):
            reward = np.maximum(0., (self.strike - price))
            discounted_reward = np.exp(-self.r * t) * reward
            done = True
            print('Option at maturity')
            return np.array((price, t)), reward, discounted_reward, done, {}
        else:

            if action == 1: #exercise the option
                reward = np.maximum(0., (self.strike - price))
                discounted_reward = np.exp(-self.r * t) * reward
                done = True
                print('Option exercised before expiration') # should always be > 0 if the agent has learnt
                return np.array((price, t)), reward, discounted_reward, done, {}
            else:
                reward = 0.
                done = False
                dt = self.fraction_list[ind] - t
                eps = np.random.normal(0, 1)
                self.state = price \
                        * np.exp((self.r - self.sigma ** 2 / 2) * dt + self.sigma * np.sqrt(dt) * eps), \
                t + dt
                self.history_states.append(self.state)
                return np.array(self.state), reward, reward, done, {}


    def isterminal(self, t):
        if t == self.fraction_list[-1]:
            return True

    def reset(self):
        self.state = self.initial_price, self.fraction_list[0]
        self.history_states = [self.state]
        return np.array(self.state)

    def render(self):
        #plt.plot(np.array(self.history_states)[:, 1], np.array(self.history_states)[:, 0])
        plt.plot(self.time_grid[:len(self.history_states)], np.array(self.history_states)[:, 0])

        plt.show()
