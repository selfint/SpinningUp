import tensorflow as tf
from tensorflow.keras import layers
from q_learning import QLearner
from gym.spaces import Discrete, Box
from typing import Union


class DeepQLearner(QLearner):
    def __init__(
        self, action_space: Union[Discrete, Box], observation_space: Union[Discrete, Box], **hyper_params
    ):
        super().__init__(action_space, observation_space)

        # no need for q table
        del self.q_table

        # generate neural network instead of q table
        self.q_network = tf.keras.Sequential([layers.Dense(24, activation="relu")])


if __name__ == "__main__":
    dq = DeepQLearner(Discrete(1), Discrete(1))
