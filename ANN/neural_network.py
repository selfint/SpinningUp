import numpy as np
from typing import List, Union, Dict, Tuple
from dense_layer import Dense, Layer
from cost_functions import COST_FUNCTIONS


class NeuralNetwork:
    def __init__(self, cost_function: str = "mse", learning_rate: float = 0.01):
        """Generate a base neural network that can store layers and use them
        to make predictions on input data, and train them on test data
        
        Keyword Arguments:
            cost_function {str} -- how to evaluate the error of the network (default: {"mse"})
            learning_rate {float} -- step size when training the network (default: {0.01})
        
        Raises:
            KeyError: only accepts cost_function defined in cost_functions.py
        """
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
                self.layers[-1].delta_neuron_activations = self.cost_function(
                    network_prediction, target, deriv=True
                )
            else:
                next_layer = self.layers[layer_index + 1]

                # the gradient of the cost function relative the activations of neurons in this layer
                # are relative to the next layer, see Dense.calculate_delta_neuron_activations for more help
                layer.calculate_delta_neuron_activations(next_layer)

        # calculate the change to each weight and bias in each layer
        previous_layer_activations = row
        for layer_index, layer in enumerate(self.layers):
            layer.calculate_delta_weights_biases(previous_layer_activations)
            previous_layer_activations = layer.neuron_activations

        # return network deltas - the changes to each weight and bias in the network
        network_delta_weights_and_biases = {
            layer: (layer.delta_weights, layer.delta_biases) for layer in self.layers
        }
        return network_delta_weights_and_biases

    def train(
        self,
        input_data: List[np.array],
        target_data: List[np.array],
        epochs: int,
        verbose: bool = False,
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

                # get the changes to the weights and biases for this row and target in the training data
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

            # apply changes, scaled by learning_rate
            for layer in self.layers:
                layer.weights -= avg_updates[layer][0] * self.learning_rate
                layer.biases -= avg_updates[layer][1] * self.learning_rate

            if verbose and epoch % 100 == 0:
                avg_cost = sum(network_cost) / len(network_cost)
                print(f"epoch {epoch}: loss={avg_cost}")

if __name__ == "__main__":
    n = NeuralNetwork(cost_function="mse", learning_rate=0.01)
    n.append_layer(Dense(size=10, input_dimensions=28 * 28, activation="lrelu"))
    n.append_layer(Dense(size=10, input_dimensions=10, activation="lrelu"))
    n.append_layer(Dense(size=10, input_dimensions=10, activation="sigmoid"))
