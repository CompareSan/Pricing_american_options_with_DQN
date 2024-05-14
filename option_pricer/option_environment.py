import copy
import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class OptionEnvironment: # Extend Option env to inherit from gym.Env base class. Then change main Option.
    def __init__(
        self,
        r: float,
        sigma: float,
        strike: float,
        initial_price: float,
        today_date: dt.datetime,
        maturity_date: dt.datetime,
        freq: str,
    ):
        self.r = r
        self.sigma = sigma
        self.strike = strike
        self.initial_price = initial_price
        self.day_count = 365.0
        self.time_grid = pd.date_range(
            start=today_date,
            end=maturity_date,
            freq=freq,
        ).to_pydatetime()
        start = self.time_grid[0]
        self.fraction_list = [
            (date - start).days / self.day_count for date in self.time_grid
        ]  # Fractions of years for discount

        self.state = self.initial_price, self.fraction_list[0]
        self._observation_space = np.array(copy.deepcopy(self.state))
        self._action_space = AttrDict()
        self._action_space.update({"n": 2})
        self.history_states = [self.state]

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def step(self, action, ind):
        price, t = self.state
        if self.isterminal(t):
            reward = np.maximum(0.0, (self.strike - price))
            discounted_reward = np.exp(-self.r * t) * reward
            done = True
            print("Option at maturity")
            return np.array((price, t)), reward, discounted_reward, done, {}
        else:

            if action == 1:  # exercise the option
                reward = np.maximum(0.0, (self.strike - price))
                discounted_reward = np.exp(-self.r * t) * reward
                done = True
                print("Option exercised before expiration")  # should  be > 0 if the agent has learnt
                return np.array((price, t)), reward, discounted_reward, done, {}
            else:
                reward = 0.0
                done = False
                dt = self.fraction_list[ind] - t
                eps = np.random.normal(0, 1)
                self.state = price * np.exp((self.r - self.sigma**2 / 2) * dt + self.sigma * np.sqrt(dt) * eps), t + dt
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
        plt.plot(self.time_grid[: len(self.history_states)], np.array(self.history_states)[:, 0], "-o")

        plt.show()
