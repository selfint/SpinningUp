from typing import List, Union

import numpy as np
import tensorflow as tf
from gym.spaces import Box
from tensorflow.keras import layers

from q_learning import Agent, Discrete, QLearner, Transition

DEFAULTS = {
    "q_network_hidden": [24, 24],
    "activation": "relu",
    "output_activation": "softmax",
    "loss": "mse",
    "optimizer": "adam",
}


class DeepQLearner(QLearner):

    # types for vscode
    action_space: Union[Discrete, Box]
    observation_space: Union[Discrete, Box]
    epsilon_decay: float
    epsilon: float
    gamma: float
    alpha: float
    buffer_size: int
    replay_buffer: List[Transition]

    q_network_hidden: List[int]
    activation: str
    loss: str
    optimizer: str
    output_activation: str

    def __init__(
        self,
        action_space: Union[Discrete, Box],
        observation_space: Union[Discrete, Box],
        **hyper_params: dict,
    ):
        # add defaults to hyper params if fields are missing
        DEFAULTS.update(hyper_params)
        super().__init__(action_space, observation_space, **DEFAULTS)

    def generate_q_table(self):
        assert isinstance(self.action_space, Discrete) or isinstance(
            self.action_space, Box
        ), "Only discrete action spaces supported. "
        assert isinstance(self.observation_space, Discrete) or isinstance(
            self.observation_space, Box
        ), "Only discrete observation spaces supported. "

        # get network iget_best_action      
        network_input_dim = Agent.get_space_size(self.observation_space)
        network_output_dim = Agent.get_space_size(self.action_space)

        # add first hidden layer
        network_layers = [
            layers.Dense(
                self.q_network_hidden[0],
                input_shape=(network_input_dim,),
                activation=self.activation,
            )
        ]

        # add the rest of the hidden layers
        for dim in self.q_network_hidden[1:]:
            network_layers.append(layers.Dense(dim, activation=self.activation))

        # add softmax output layer
        network_layers.append(layers.Dense(network_output_dim, activation=self.output_activation))

        # set q table to be a neural network
        self.q_table = tf.keras.Sequential(network_layers)
        self.q_table.compile(loss=self.loss, optimizer=self.optimizer)

    def get_best_action(self, observation) -> List[float]:
        if isinstance(self.action_space, Discrete):
            return np.armgax(self.q_table(observation))
        elif isinstance(self.action_space, Box):
            return self.q_table(observation)

    def learn(self, observation, action, reward, next_observation):
        """
        Train the neural network using replay buffer
        """


if __name__ == "__main__":
    obs_space = Box(low=np.array([-1]), high=np.array([1]))
    dq = DeepQLearner(obs_space, Discrete(1), output_activation="sigmoid")
    print(dq.act(obs_space.sample()))
