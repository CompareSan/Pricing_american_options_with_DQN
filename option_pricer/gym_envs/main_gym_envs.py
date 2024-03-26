import gymnasium as gym
import torch
from option_pricer.agent import DQNAgent
from option_pricer.neural_network import DQN
from option_pricer.reply_buffer import ReplyBuffer


def main(agent: DQNAgent, env: gym.Env, n_episodes: int, model_name: str):
    scores = []
    counter = 0
    for i in range(n_episodes):
        done = False
        score = 0
        observation, _ = env.reset()

        while not done:
            action = agent.take_action(observation)
            observation_, reward, terminated, truncated, _ = env.step(action)
            score += reward
            done = terminated or truncated
            agent.store(observation, action, reward, observation_, done)
            observation = observation_
            agent.train(counter)
            counter += 1

        scores.append(score)

        print("episode: {}, score: {}".format(i, score))
        print(agent.epsilon)

    agent.save_model(f"{model_name}.pt")


if __name__ == "__main__":
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu",
    )
    env_name = "LunarLander-v2"
    env = gym.make(env_name, render_mode="human")
    mem_size = 10000
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

    n_episodes = 400
    initial_epsilon = 1
    final_epsilon = 0
    epsilon_decay = (initial_epsilon - final_epsilon) / 25000
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
    main(agent=agent, env=env, n_episodes=n_episodes, model_name=env_name)
