#import gym
import numpy as np
from agent import Agent
from environment import OptionEnvironment
import datetime as dt

#env = gym.make("MountainCar-v0")
start_date = dt.datetime(2020, 1, 1)
maturity_date = dt.datetime(2021, 12, 31)

env = OptionEnvironment(r=0.06, sigma=0.2, strike=40, initial_price=44.,
                            today_date=start_date, maturity_date=maturity_date, freq='W')


n_episodes = 5000
agent = Agent(gamma=1, epsilon=1, alpha=0.001, input_dims=2, n_actions=2,
              mem_size=1000000, batch_size=64, epsilon_end=0.01)

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
        action = agent.choose_action(observation)
        observation_, reward, discounted_reward, done, info = env.step(action, ind)
        #env.render()
        score += reward
        discounted_score += discounted_reward
        agent.remember(observation, action, reward, observation_, done)
        observation = observation_
        agent.learn(counter)
        counter += 1
        ind += 1
    #if i > 50:
        #env.render()
    eps.append(agent.epsilon)
    scores.append(score)
    discounted_scores.append(discounted_score)
    print('episode: {}, score: {}'.format(i, score))

print("Option price: {:.2f}" .format(np.mean(discounted_scores[-1000:])))
