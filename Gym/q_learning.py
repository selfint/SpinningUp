from agent import Agent
import numpy as np


EPSILON_DECAY = 0.01


class QLearner(Agent):
    def __init__(self, action_dim, observation_dim):
        super().__init__(action_dim, observation_dim)

        # generate empty q table
        self.q_table = np.zeros(shape=(action_dim, observation_dim))
        self.epsilon = 1.0

    def act(self, observation):
        pass

    def learn(self, observation, action, reward, new_observation):
        pass
