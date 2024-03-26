# import gym
import datetime as dt

import numpy as np
import torch
from neural_network import DQN
from reply_buffer import ReplyBuffer

from option_pricer.agent import DQNAgent
from option_pricer.option_environment import OptionEnvironment


def main(agent: DQNAgent, env: OptionEnvironment, n_episodes: int):
    scores = []
    discounted_scores = []
    eps = []
    counter = 0
    for i in range(n_episodes):
        done = False
        score = 0
        discounted_score = 0
        observation = env.reset()
        ind = 1
        while not done:
            action = agent.take_action(observation)
            observation_, reward, discounted_reward, done, info = env.step(
                action,
                ind,
            )
            score += reward
            discounted_score += discounted_reward
            agent.store(
                observation,
                action,
                discounted_reward,  # try discounted reward
                observation_,
                done,
            )
            observation = observation_
            agent.train(counter)
            counter += 1
            ind += 1
        # if i % 1000 == 0:
        # env.render()
        eps.append(agent.epsilon)

        scores.append(score)
        discounted_scores.append(discounted_score)
        print("episode: {}, score: {}".format(i, score))
        print(f"eps: {eps[-1]}")
        if len(discounted_scores) >= 1000:
            print("Option price: {:.2f}".format(np.mean(discounted_scores[-100:])))


if __name__ == "__main__":
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu",
    )
    start_date = dt.datetime(2020, 1, 1)
    maturity_date = dt.datetime(2020, 12, 1)

    env = OptionEnvironment(
        r=0.06,
        sigma=0.1,
        strike=40,
        initial_price=41,
        today_date=start_date,
        maturity_date=maturity_date,
        freq="M",
    )

    mem_size = 50000
    batch_size = 128
    replay_buffer = ReplyBuffer(
        max_size=mem_size,
        batch_size=batch_size,
        state_shape=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        discrete=True,
    )

    q_network = DQN(
        env.observation_space.shape[0],
        env.action_space.n,
    )

    alpha = 3e-4
    optimizer = torch.optim.AdamW(
        q_network.parameters(),
        lr=alpha,
        amsgrad=True,
    )

    initial_epsilon = 1
    final_epsilon = 0.01
    epsilon_decay = (initial_epsilon - final_epsilon) / 50000
    n_episodes = 20000
    agent = DQNAgent(
        env=env,
        replay_buffer=replay_buffer,
        q_network=q_network,
        initial_epsilon=initial_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        optimizer=optimizer,
        device=device,
    )
    main(agent=agent, env=env, n_episodes=n_episodes)
