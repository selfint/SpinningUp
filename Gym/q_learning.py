from agent import Agent
import numpy as np


EPSILON_DECAY = 0.01


class QLearner(Agent):
    def __init__(
        self, action_dim, observation_dim, action_space_type, observation_space_type
    ):
        super().__init__(
            action_dim, observation_dim, action_space_type, observation_space_type
        )

        # only support discrete actions and obsercations
        # TODO: add continuous space support
        assert (
            action_space_type == "discrete" and observation_space_type == "discrete"
        ), "Only discrete action and observation spaces supported"

        # generate empty q table
        self.q_table = np.zeros(shape=(action_dim, observation_dim))
        self.epsilon = 1.0

    def act(self, observation):
        """act according to an epsilon-greedy strategy using 
        the agents q table
        """

        if np.random.rand() < self.epsilon:
            # TODO: take random action
            pass
        else:
            # TODO: take action with max q value
            pass

    def learn(self, observation, action, reward, new_observation):
        """update q table using reward
        """
        pass
