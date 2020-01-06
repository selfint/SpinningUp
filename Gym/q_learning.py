from agent import Agent
import numpy as np
from gym.spaces import Discrete


EPSILON_DECAY = 0.9999  # rate at which agent stops taking random actions
GAMMA = 0.999  # discount rate, how much are future rewards important
ALPHA = 0.05  # learning rate, how much does one sample influence agent


class QLearner(Agent):
    def __init__(self, action_space: Discrete, observation_space: Discrete):
        super().__init__(action_space, observation_space)

        # only support discrete actions and obsercations
        # TODO: add continuous space support
        assert isinstance(action_space, Discrete) and isinstance(
            observation_space, Discrete
        ), "Only discrete action and observation spaces supported"

        # generate empty q table
        self.action_space = action_space
        self.observation_space = observation_space
        self.q_table = np.zeros(shape=(self.observation_space.n, self.action_space.n))
        self.epsilon = 1.0

    def act(self, observation):
        """act according to an epsilon-greedy strategy using 
        the agents q table
        """
        self.epsilon *= EPSILON_DECAY

        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.q_table[observation])

    def learn(self, observation, action, reward, new_observation):
        """update q table using reward, and decrease epsilon
        """
        self.q_table[observation][action] = (1 - ALPHA) * self.q_table[observation][action] + ALPHA * (
            reward + np.max(self.q_table[new_observation])
        )
