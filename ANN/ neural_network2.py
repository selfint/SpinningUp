import numpy as np
from typing import List, Union, Dict, Tuple
from dense_layer import Dense
from layer import Layer
from cost_functions import COST_FUNCTIONS


class NeuralNetwork:
    def __init__(self, cost_function: str = "mse", learning_rate: float = 0.01):
        self.layers: List[Layer] = []
        try:
            self.cost_function = COST_FUNCTIONS[cost_function]
        except KeyError:
            raise KeyError(
                f"Cost function function {cost_function} is not defined in cost_functions.py file"
            )

        self.learning_rate = learning_rate

    def append_layer(self, layer: Layer) -> None:
        """Append a layer to the network
        
        Arguments:
            layer {Layer} -- layer to append
        """
        self.layers.append(layer)

    def predict(self, inputs: np.array) -> np.array:
        """Get the network prediction using forward propagation
        through all layers in the network
        
        Arguments:
            inputs {np.array} -- network input
        
        Returns:
            np.array -- network output
        """
        layer_input = inputs
        for layer in self.layers:
            layer_input = layer.feed_forward(layer_input)
        return layer_input

    def calculate_network_delta_weights_and_biases(
        self, row: np.array, target: np.array
    ) -> Dict[Layer, Tuple[np.array, np.array]]:
        """Propagate the network error using back propagations and use the errors
        to calculate the changes for each weight and bias
        
        Arguments:
            row {np.array} -- network input
            target {np.array} -- target network output
        
        Returns:
            Dict[Layer, Tuple[np.array, np.array]] -- Changes to the weights and biases in each layer in the network
        """

        # feed forward row data through the network
        network_prediction = self.predict(row)

        # add a fake 'layer' to represent the network inputs
        layer_activations = [row] + [layer.neuron_activations for layer in self.layers]

        # propagate the gradient backwards and find all gradients of the cost function with respect to
        # the activation of each neuron
        for reversed_layer_index, layer in enumerate(reversed(self.layers)):
            layer_index = len(self.layers) - 1 - reversed_layer_index

            # manually calculate the gradient of the cost function relative to the output
            # of the last layers activations using the actual gradient formula of the cost function
            if layer_index == len(self.layers) - 1:
                self.layers[-1].delta_neuron_activations = self.cost_function(network_prediction, target, deriv=True)
            else:
                next_layer = self.layers[layer_index + 1]
                layer.calculate_delta_neuron_activations(next_layer)

        # calculate the change to each weight and bias
        previous_layer_activations = row
        for layer_index, layer in enumerate(self.layers):
            layer.calculate_delta_weights_biases(previous_layer_activations)
            previous_layer_activations = layer.neuron_activations

        # return network deltas
        network_delta_weights_and_biases = {
            layer: (layer.delta_weights, layer.delta_biases) for layer in self.layers
        }
        return network_delta_weights_and_biases

    def train(
        self, input_data: List[np.array], target_data: List[np.array], epochs: int, verbose: bool = False
    ) -> None:
        """Train the network using back propagation
        
        Arguments:
            input_data {List[np.array]} -- training data
            target_data {List[np.array]} -- training data labels
            epochs {int} -- how many epochs to train
            verbose {bool} -- print training progress
        """
        for epoch in range(epochs):

            if verbose:
                network_cost = []

            # save all the updates for each layer for each training sample
            network_updates: List[dict] = []
            for row, target in zip(input_data, target_data):
                if verbose:
                    network_cost.append(self.cost_function(self.predict(row), target))

                network_delta_weights_and_biases = self.calculate_network_delta_weights_and_biases(
                    row, target
                )
                network_updates.append(network_delta_weights_and_biases)

            # average all changes to each weight and bias and update the network
            # TODO: make this more efficient
            sum_updates = network_updates[0]
            for network_update in network_updates[1:]:
                for layer, (delta_weights, delta_biases) in network_update.items():
                    sum_updates[layer] = (
                        sum_updates[layer][0] + delta_weights,
                        sum_updates[layer][1] + delta_biases,
                    )

            avg_updates = sum_updates
            for delta_weights, delta_biases in avg_updates.values():
                delta_weights /= len(network_updates)
                delta_biases /= len(network_updates)

            # apply changes
            for layer in self.layers:
                layer.weights -= avg_updates[layer][0] * self.learning_rate
                layer.biases -= avg_updates[layer][1] * self.learning_rate

            if verbose and epoch % 100 == 0:
                avg_cost = sum(network_cost) / len(network_cost)
                print(f"epoch {epoch}: loss={avg_cost}")


if __name__ == "__main__":
    n = NeuralNetwork(cost_function="mse", learning_rate=0.01)
    n.append_layer(Dense(size=4, input_dimensions=2, activation="relu"))
    n.append_layer(Dense(size=1, input_dimensions=4, activation="sigmoid"))

    xor_train = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    xor_test = np.array([[0], [1], [1], [0]])
    n.train(xor_train, xor_test, 20000, verbose=True)
    print(n.predict(xor_train[0]))
    print(n.predict(xor_train[1]))
    print(n.predict(xor_train[2]))
    print(n.predict(xor_train[3]))
