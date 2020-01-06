import gym
from agent import Agent
from q_learning import QLearner
import matplotlib.pyplot as plt
import numpy as np

env_name = "Taxi-v2"
env = gym.make(env_name)
observation = env.reset()
agent: QLearner = QLearner(env.action_space, env.observation_space)

total_reward = 0
rewards = []
avg_rewards = []
epsilons = []
for _ in range(300000):
    # env.render()
    action = agent.act(observation)
    new_observation, reward, done, info = env.step(action)
    agent.learn(observation, action, reward, new_observation)
    observation = new_observation
    total_reward += reward

    if done:
        observation = env.reset()
        rewards.append(total_reward)
        avg_rewards.append(np.average(rewards))
        epsilons.append(agent.epsilon * 10)
        total_reward = 0

if input("play? ") == "y":
    observation = env.reset()
    done = False
    while not done:
        action = agent.act(observation)
        observation, _, done, _ = env.step(action)
        env.render()
        print(reward)
        if input("continue? ") == "n":
            break
env.close()
print(agent.q_table)
plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot(epsilons)
plt.show()


