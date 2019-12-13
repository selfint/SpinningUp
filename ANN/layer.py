import numpy as np

from typing import List
from activations import ACTIVATIONS


class Layer:
    def __init__(
        self,
        size: int,
        input_dimensions: int,
        activation: str = "sigmoid",
        weight_range: float = 1.0,
        bias_range: float = 1.0,
    ):

        self.size = size
        self.input_dimensions = input_dimensions
        try:
            self.activation = ACTIVATIONS[activation]
        except KeyError:
            raise KeyError(
                f"Activation function {activation} is not defined in activations.py file"
            )

        self.neuron_inputs = np.zeros(size)
        self.neuron_activations = np.zeros(size)

        # these gradients represent the uphill direction of the cost function with respect to each, (respectively :P)
        self.delta_neuron_activations = np.zeros(size)  # the gradient of the cost function respective to the neuron activaitons
        self.delta_weights = np.zeros((size, input_dimensions))  # the gradient of the cost function respective to the weights
        self.delta_biases = np.zeros(size)  # the gradient of the cost function respective to the biases
        self.weights = np.random.uniform(
            size=(size, input_dimensions), low=-weight_range, high=weight_range
        )
        self.biases = np.random.uniform(size=size, low=-bias_range, high=bias_range)

    def feed_forward(self, inputs: np.array) -> np.array:
        """Feeds forward the given input and returns the
        neuron activations
 
        Arguments:
            inputs {np.array} -- layer input

        Returns:
            np.array -- layer activations
        """
        raise NotImplementedError()

    def calculate_delta_neuron_activations(self, next_layer) -> None:
        """Calculates the gradient of the cost of the neurons activation using the
        gradients of the cost of the neurons in the next layer
        
        Arguments:
            next_layer {Layer} -- Next layer in the network, if there is no next layer
            then the neuron errors must be calculated using the derivative of the cost function
        """
        raise NotImplementedError()
