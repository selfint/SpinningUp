from random import choice
from typing import List

import numpy as np
from gym.spaces import Discrete

from agent import Agent, Transition


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

        # get hyper params
        for param_name, default_value in DEFAULTS.items():
            self.get_hyper_param(param_name, default_value)

        # report param values
        vals = "\n" + "\n".join(
            [f"{param_name}={getattr(self, param_name)}" for param_name in DEFAULTS]
        )
        print(f"Parameter values: {vals}")

        # init replay buffer
        self.replay_buffer: List[Transition] = []

    def act(self, observation):
        """
        act according to an epsilon-greedy strategy using 
        the agents q table
        """
        self.epsilon *= self.epsilon_decay

        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.get_action_values(observation))

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
            ][t_action] + self.alpha * (t_reward + np.max(self.q_table[t_next_observation]))

    def get_action_values(self, observation) -> List[float]:
        return self.q_table[observation]


if __name__ == "__main__":
    q = QLearner(Discrete(4), Discrete(4))
