import datetime as dt

import torch
from option_pricer.neural_network import Policy

from option_pricer.agent import Agent, DQNAgent, ReinforceAgent
from option_pricer.option_environment import OptionEnvironment


def main(agent: Agent, n_episodes: int, model_name: str):
    agent.train(n_episodes)
    agent.save_model(f"{model_name}.pt")


if __name__ == "__main__":
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu",
    )
    start_date = dt.datetime(2020, 1, 1)
    maturity_date = dt.datetime(2020, 12, 1)
    env_name = "Option-Pricer-v2"
    env = OptionEnvironment(
        r=0.06,
        sigma=0.7,
        strike=40,
        initial_price=38,
        today_date=start_date,
        maturity_date=maturity_date,
        freq="D",
    )
    
    policy_network = Policy(
        env.observation_space.shape[0],
        env.action_space.n,
    )

    alpha = 3e-4
    optimizer = torch.optim.AdamW(
        policy_network.parameters(),
        lr=alpha,
        amsgrad=True,
    )
    n_episodes = 40000

    agent = DQNAgent(
        env=env,
        policy_network=policy_network,
        optimizer=optimizer,
        device=device,
    )

    main(agent=agent, n_episodes=n_episodes, model_name=env_name)
