from random import choice
from typing import List

import numpy as np

from agent import Agent, Transition
from gym.spaces import Discrete

DEFAULTS = {
    "epsilon_decay": 0.9999,  # rate at which agent stops taking random actions
    "gamma": 0.999,  # discount rate, how much are future rewards important
    "alpha": 0.05,  # learning rate, how much does one sample influence agent
    "buffer_size": 100,  # size of replay buffer
}


class QLearner(Agent):

    # types for vscode
    action_space: Discrete
    observation_space: Discrete
    epsilon_decay: float
    epsilon: float
    gamma: float
    alpha: float
    buffer_size: int
    replay_buffer: List[Transition]

    def __init__(
        self, action_space: Discrete, observation_space: Discrete, **hyper_params
    ):
        DEFAULTS.update(hyper_params)
        super().__init__(action_space, observation_space, **DEFAULTS)
        self.epsilon = 1.0

        # init replay buffer
        self.replay_buffer: List[Transition] = []

        # init q table
        self.q_table = self.generate_q_table()

    def act(self, observation):
        """
        act according to an epsilon-greedy strategy using 
        the agents q table
        """
        self.epsilon *= self.epsilon_decay

        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        else:
            return self.get_best_action(observation)

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
            self.q_table[t_observation][t_action] = (1 - self.alpha) * self.q_table[
                t_observation
            ][t_action] + self.alpha * (
                t_reward + np.max(self.q_table[t_next_observation])
            )

    def get_best_action(self, observation) -> List[float]:
        return np.argmax(self.q_table[observation])

    def generate_q_table(self) -> None:
        # only support discrete actions and obsercations
        # TODO: add continuous space support
        assert isinstance(
            self.action_space, Discrete
        ), "Only discrete action spaces supported. "
        assert isinstance(
            self.observation_space, Discrete
        ), "Only discrete observation spaces supported. "

        # generate empty q table
        return np.zeros(shape=(self.observation_space.n, self.action_space.n))


if __name__ == "__main__":
    q = QLearner(Discrete(4), Discrete(4))
