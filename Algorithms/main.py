import gym
import matplotlib.pyplot as plt
import numpy as np

from agent import Agent
from q_learning import QLearner

env_name = "Taxi-v2"
env = gym.make(env_name)
observation = env.reset()
agent = QLearner(
    env.action_space,
    env.observation_space,
    epislon_decay=0.9999,
    alpha=0.0001,
    buffer_size=50,
    gamma=0.99,
)
epochs = 30000

total_reward = 0
rewards = []
avg_rewards = []
epsilons = []
for _ in range(epochs):
    # env.render()
    action = agent.act(observation)
    next_observation, reward, done, info = env.step(action)
    agent.learn(observation, action, reward, next_observation)
    observation = next_observation
    total_reward += reward

    if done:
        observation = env.reset()
        rewards.append(total_reward)
        avg_rewards.append(np.average(rewards[-50:]))
        epsilons.append(-agent.epsilon * 800)
        total_reward = 0
env.close()

print(agent.q_table)
plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot(epsilons)
plt.show()
