import gymnasium as gym
import torch

from option_pricer.agent import Agent, DQNAgent, ReinforceAgent
from option_pricer.neural_network import Policy


def main(agent: Agent, n_episodes: int, model_name: str):
    agent.train(n_episodes)
    agent.save_model(f"{model_name}.pt")


if __name__ == "__main__":
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu",
    )
    env_name = "LunarLander-v2"
    env = gym.make(env_name, render_mode="human")

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
    n_episodes = 400

    agent = DQNAgent(
        env=env,
        policy_network=policy_network,
        optimizer=optimizer,
        device=device,
    ) # Change to DQNAgent for DQN

    main(agent=agent, n_episodes=n_episodes, model_name=env_name)
