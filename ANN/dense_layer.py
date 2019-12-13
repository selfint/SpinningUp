import numpy as np

from typing import List
from layer import Layer


class Dense(Layer):
    def __init__(
        self,
        size: int,
        input_dimensions: int,
        activation: str = "sigmoid",
        weight_range: float = 1.0,
        bias_range: float = 1.0,
    ):
        super().__init__(size, input_dimensions, activation, weight_range, bias_range)

    def feed_forward(self, inputs: np.array) -> np.array:
        for neuron_index in range(self.size):
            neuron_input = sum(
                self.weights[neuron_index] * inputs + self.biases[neuron_index]
            )
            self.neuron_inputs[neuron_index] = neuron_input

            neuron_activation = self.activation(neuron_input)
            self.neuron_activations[neuron_index] = neuron_activation

        return self.neuron_activations

    def calculate_delta_neuron_activations(self, next_layer: Layer) -> None:
        self.delta_neuron_activations = np.zeros(self.size)
        for next_neuron_index, next_neuron_delta_activation in enumerate(
            next_layer.delta_neuron_activations
        ):
            self.delta_neuron_activations += (
                next_neuron_delta_activation
                * next_layer.activation(
                    next_layer.neuron_inputs[next_neuron_index], deriv=True
                )
                * next_layer.weights[next_neuron_index]
            )

    def calculate_delta_weights_biases(self, previous_layer_activations: np.array):
        self.delta_weights = np.zeros((self.size, self.input_dimensions))
        for neuron_index in range(self.size):

            # the gradient of the cost with respect to each weight, using the chain rule is:
            #     - the gradient of the cost function with respect to the neuron's activation
            #     - multiplied by the gradient of the neuron's activation with respect to the neurons input
            #     - multiplied by the gradient of the neuron's input with respect to the weight
            self.delta_weights[neuron_index] = (
                self.delta_neuron_activations[
                    neuron_index
                ]  # gradient of the cost of the activation of the current neuron
                * self.activation(
                    self.neuron_inputs[neuron_index], deriv=False
                )  # gradient of the activation of the current neuron
                * previous_layer_activations  # activations of the previous layer
            )

        # the gradient of the cost with respect to each bias, using the chain rule is:
        #     - the gradient of the cost function with respect to the neuron's activation
        #     - multiplied by the gradient of the neuron's activation with respect to the neurons input
        #     - multiplied by the gradient of the neuron's input with respect to the bias
        #     * the gradient of the neuron's input with respect to the bias = 1, since the bias is just added
        #       to the input (see feed_forward function for the calculation of the neuron input)
        self.delta_biases = (
            self.delta_neuron_activations  # gradient of the cost of the activation of the current neuron
            # map the activation function over the neurons inputs
            # vectorize turns a normal function into one that can handle vectors
            # (maybe it does more but thats how i use it here)
            * np.vectorize(self.activation)(
                self.neuron_inputs, deriv=False
            )  # gradient of each neuron's activation with respect to its input
        )


if __name__ == "__main__":
    d = Dense(size=3, input_dimensions=1)
    d2 = Dense(size=2, input_dimensions=3)
    d2.feed_forward(d.feed_forward(np.array([1])))

    d2.delta_neuron_activations = np.array([1, 2])
    d2.calculate_delta_weights_biases(d.neuron_activations)
