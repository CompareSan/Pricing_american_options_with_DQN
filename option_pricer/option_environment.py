import datetime as dt
from typing import Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class OptionEnvironment(gym.Env):
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
        super(OptionEnvironment, self).__init__()
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
        self.history_states = [self.state]

        self.action_space = gym.spaces.Discrete(
            2
        )  # 2 discrete actions: exercise or hold
        self.observation_space = gym.spaces.Box(
            low=0, high=np.inf, shape=(2,), dtype=np.float32
        )  # price and time fraction

    def step(self, action):
        price, t = self.state
        ind = self.fraction_list.index(t)
        if self.isterminal(t):
            reward = np.maximum(0.0, (self.strike - price))
            discounted_reward = np.exp(-self.r * t) * reward
            done = True
            print("Option at maturity")
            return (
                np.array((price, t), dtype=np.float32),
                discounted_reward,
                done,
                False,
                None,
            )
        else:
            if action == 1:  # exercise the option
                reward = np.maximum(0.0, (self.strike - price))
                discounted_reward = np.exp(-self.r * t) * reward
                done = True
                print(
                    "Option exercised before expiration"
                )  # should be > 0 if the agent has learnt
                return (
                    np.array((price, t), dtype=np.float32),
                    discounted_reward,
                    done,
                    False,
                    None,
                )
            else:
                reward = 0.0
                done = False
                dt = self.fraction_list[ind + 1] - t
                eps = np.random.normal(0, 1)
                self.state = (
                    price
                    * np.exp(
                        (self.r - self.sigma**2 / 2) * dt
                        + self.sigma * np.sqrt(dt) * eps
                    ),
                    self.fraction_list[ind + 1],
                )
                self.history_states.append(self.state)
                return (
                    np.array(self.state, dtype=np.float32),
                    reward,
                    done,
                    False,
                    None,
                )

    def isterminal(self, t):
        if t == self.fraction_list[-1]:
            return True

    def reset(self):
        self.state = self.initial_price, self.fraction_list[0]
        self.history_states = [self.state]
        return np.array(self.state, dtype=np.float32), None

    def render(self, mode: Optional[str] = None):
        plt.plot(
            self.time_grid[: len(self.history_states)],
            np.array(self.history_states)[:, 0],
            "-o",
        )
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.title("Option Price Evolution")
        plt.show()
