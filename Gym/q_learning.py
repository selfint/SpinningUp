from random import choice
from typing import List

import numpy as np
from gym.spaces import Discrete

from agent import Agent, Transition

EPSILON_DECAY = 0.9999  # rate at which agent stops taking random actions
GAMMA = 0.999  # discount rate, how much are future rewards important
ALPHA = 0.05  # learning rate, how much does one sample influence agent
BUFFER_SIZE = 100  # size of replay buffer


class QLearner(Agent):

    # types for vscode
    action_space: Discrete
    observation_space: Discrete
    epsilon_decay: float
    gamma: float
    alpha: float
    buffer_size: int
    replay_buffer: List[Transition]

    def __init__(
        self, action_space: Discrete, observation_space: Discrete, **hyper_params
    ):
        super().__init__(action_space, observation_space, **hyper_params)

        # only support discrete actions and obsercations
        # TODO: add continuous space support
        assert isinstance(
            action_space, Discrete
        ), "Only discrete action spaces supported. "
        assert isinstance(
            observation_space, Discrete
        ), "Only discrete observation spaces supported. "

        # generate empty q table
        self.q_table = np.zeros(shape=(self.observation_space.n, self.action_space.n))
        self.epsilon = 1.0

        # hyper params
        self.get_hyper_param("epsilon_decay", EPSILON_DECAY)
        self.get_hyper_param("gamma", GAMMA)
        self.get_hyper_param("alpha", ALPHA)
        self.get_hyper_param("buffer_size", BUFFER_SIZE)

        # init replay buffer
        self.replay_buffer: List[Transition] = []

    def act(self, observation):
        """
        act according to an epsilon-greedy strategy using 
        the agents q table
        """
        self.epsilon *= EPSILON_DECAY

        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.q_table[observation])

    def learn(self, observation, action, reward, next_observation):
        """
        update q table using reward, and decrease epsilon
        """

        # log transition in replay buffer
        transition = self.buffer_transition(
            observation, action, reward, next_observation
        )

        # choose a random amount of transitions from buffer
        random_amount = np.random.randint(0, len(self.replay_buffer))
        transitions = [choice(self.replay_buffer) for _ in range(random_amount)] + [
            transition
        ]

        # learn from transition and random transitions from the replay buffer
        for t_observation, t_action, t_reward, t_next_observation in transitions:
            self.q_table[t_observation][t_action] = (1 - ALPHA) * self.q_table[
                t_observation
            ][t_action] + ALPHA * (t_reward + np.max(self.q_table[t_next_observation]))

    def buffer_transition(
        self, observation, action, reward, next_observation
    ) -> Transition:
        """
        add transition to replay buffer, and return the last given arguments as a transition
        """
        transition = Transition(observation, action, reward, next_observation)
        self.replay_buffer.insert(0, transition)
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop()

        return transition
