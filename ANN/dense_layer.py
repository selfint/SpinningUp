import numpy as np

from layer import Layer


class Dense(Layer):
    def feed_forward(self, inputs: np.array) -> np.array:

        # calculate the input and activation for each neuron in the layer
        for neuron_index in range(self.size):

            # the input of a neuron is the weighted sum of its inputs, using the weights
            # connecting it the the previous layer, plus its bias
            neuron_input = (
                sum(self.weights[neuron_index] * inputs)  # the weighted sum
                + self.biases[neuron_index]  # the bias
            )

            # save the neuron input
            self.neuron_inputs[neuron_index] = neuron_input

            # the activation of the node is simply the output of the activation function
            # for its input
            neuron_activation = self.activation(neuron_input)

            # save the neuron activation
            self.neuron_activations[neuron_index] = neuron_activation

        return self.neuron_activations

    def calculate_delta_neuron_activations(self, next_layer: Layer) -> None:

        # reset the delta neuron activations since we will be summing
        # the delta using addition while looping through the neurons
        # in the next layer
        self.delta_neuron_activations = np.zeros(self.size)

        # get the gradient of the cost function with respect to the activation of
        # each neuron in the current layer
        for next_neuron_index, next_neuron_delta_activation in enumerate(
            next_layer.delta_neuron_activations
        ):

            # the gradient of the cost function with respect to each neuron, using the chain rule is:
            #     - the gradient of the cost function with respect to the NEXT neuron's activation
            #     - multiplied by the gradient of the NEXT neuron's activation with respect to the NEXT neuron's input
            #     - multiplied by the gradient of the NEXT neuron's input with respect to the weight connecting the
            #       CURRENT neuron to the NEXT neuron
            self.delta_neuron_activations += (
                next_neuron_delta_activation
                * next_layer.activation(
                    next_layer.neuron_inputs[next_neuron_index], deriv=True
                )
                * next_layer.weights[next_neuron_index]
            )

    def calculate_delta_weights_biases(self, previous_layer_activations: np.array):

        # get the delta weights for each neuron in the layer
        for neuron_index in range(self.size):

            # the gradient of the cost with respect to each weight, using the chain rule is:
            #     - the gradient of the cost function with respect to the neuron's activation
            #     - multiplied by the gradient of the neuron's activation with respect to the neuron's input
            #     - multiplied by the gradient of the neuron's input with respect to the weight
            self.delta_weights[neuron_index] = (
                self.delta_neuron_activations[neuron_index]
                * self.activation(self.neuron_inputs[neuron_index], deriv=False)
                * previous_layer_activations
            )

        # get the delta for each bias in the layer
        # the gradient of the cost with respect to each bias, using the chain rule is:
        #     - the gradient of the cost function with respect to the neuron's activation
        #     - multiplied by the gradient of the neuron's activation with respect to the neurons input
        #     - multiplied by the gradient of the neuron's input with respect to the bias
        #     * the gradient of the neuron's input with respect to the bias = 1, since the bias is just added
        #       to the input (see feed_forward function for the calculation of the neuron input)
        self.delta_biases = self.delta_neuron_activations * np.vectorize(
            self.activation
        )(self.neuron_inputs, deriv=False)


if __name__ == "__main__":
    d = Dense(size=3, input_dimensions=1)
    d2 = Dense(size=2, input_dimensions=3)
    d2.feed_forward(d.feed_forward(np.array([1])))
    d2.delta_neuron_activations = np.array([1, 2])
    d2.calculate_delta_weights_biases(d.neuron_activations)
    print(d2.neuron_inputs)
    print(d2.neuron_activations)
    print(d2.delta_neuron_activations)
    print(d2.delta_weights)
    print(d2.delta_biases)
