from typing import List, Union

import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(
        self,
        topology: List[int],
        weight_range: float = 1.0,
        bias_range: float = 1.0,
        activations: Union[str, List[str]] = "relu",
        learning_rate: float = 0.001,
    ) -> None:
        """Generate a neural network with randomized weights and biases
        capable of forward propagation

        Arguments:
            topology {List[int]} -- the amount of nodes in each layer, 
                                    must contain at least 2 layers
        
        Keyword Arguments:
            weight_range {float} -- min (as negative) and max (as positive) values of 
                                    weights (default: {1.0})
            bias_range {float} -- min (as negative) and max (as positive) values of 
                                  biases (default: {1.0})
            activations {Union[str, List[str]]} -- activation function for each layer, except the
                                                   input layer. set to string for one activation
                                                   function for all layers (default: {"relu"})
        """

        assert len(topology) >= 2, "Network must contain at least 2 layers"

        self.topology = topology
        self.weight_range = weight_range
        self.bias_range = bias_range
        self.activations: List[str] = list()
        self.learning_rate = learning_rate

        # input layer doesn't have an activation
        if isinstance(activations, str):
            self.activations = [activations for layer in self.topology[1:]]
        else:
            self.activations = activations

        # generate weights and biases
        self.weights: List[np.array] = self._generate_weights(topology)
        self.biases: List[np.array] = self._generate_biases(topology)

    def _generate_weights(self, topology: List[int]) -> List[np.array]:
        """Generate random weights for the network

        Arguments:
            topology {List[int]} -- structure of the network

        Returns:
            List[float] -- weights to the next node
        """
        weights = list()
        for previous_index, layer in enumerate(self.topology[1:]):
            weights.append(
                np.random.random(size=(layer, self.topology[previous_index]))
                * self.weight_range
                * 2
                - self.weight_range
            )
        return weights

    def _generate_biases(self, topology: List[int]) -> List[np.array]:
        """Generate the biases for each node in the network, except the input
        nodes which don't have a bias

        Arguments:
            topology {List[int]} -- structure of the network

        Returns:
            List[float] -- biases of the network
        """

        biases = []
        for layer in self.topology[1:]:
            biases.append(
                np.random.random(layer) * self.bias_range * 2 - self.bias_range
            )

        return biases

    def _activation_function(
        self, x: float, function: str = "relu", deriv: bool = False
    ) -> float:
        """activation function for the network

        Arguments:
            x {float} -- input
 
        Keyword Arguments:
            function {str} -- which  (default: {relu})
            deriv {bool} -- use the derivative of the activation function instead (default: {False})
        
        Returns:
            float -- output
        """
        if function == "relu":
            if deriv:
                return 0 if x < 0 else 1
            return max(0, x)

        elif function == "sigmoid":
            if deriv:
                dx = 1.0 / (1.0 + np.exp(-x))
                return dx * (1 - dx)
            return 1.0 / (1.0 + np.exp(-x))
        else:
            raise NotImplementedError(f"Function {function} not implemented")

    def feed_forward(self, inputs: np.array) -> np.array:
        """Outputs the network output using feed forward propagation

        Arguments:
            inputs {np.array} -- network inputs

        Returns:
            np.array -- network output
        """
        previous_layer_output = inputs
        for layer_biases, layer_weights, layer_activation in zip(
            self.biases, self.weights, self.activations
        ):
            layer_output = []
            for node_bias, node_weights in zip(layer_biases, layer_weights):

                # apply the activation function on the output of the previous layer and add the node bias
                layer_output.append(
                    self._activation_function(
                        self._transfer_function(
                            previous_layer_output, node_bias, node_weights
                        ),
                        function=layer_activation,
                    )
                )
            previous_layer_output = np.array(layer_output)
        return previous_layer_output

    def _transfer_function(
        self, previous_layer_output: np.array, node_bias: float, node_weights: np.array
    ) -> float:
        """transfer function that calculates the input for a given node in the network
        
        Arguments:
            previous_layer_output {np.arry} -- output of the previous layer
            node_bias {float} -- bias of the node
            node_weights {np.array} -- weights connecting the previous layer to the node
        
        Returns:
            float -- node input
        """
        return sum(previous_layer_output * node_weights) + node_bias

    def _get_layer_outputs(self, inputs: np.array) -> List[np.array]:
        """Returns the output of each layer during the feed forward propagation
        
        Arguments:
            inputs {np.array} -- network inputs
        
        Returns:
            List[np.array] -- output of each layer in the network
        """
        # ignore input layer outputs - can't be changed in training
        layer_outputs = []

        previous_layer_output = inputs
        for layer_biases, layer_weights, layer_activation in zip(
            self.biases, self.weights, self.activations
        ):
            layer_output = []
            for node_bias, node_weights in zip(layer_biases, layer_weights):

                # apply the activation function on the output of the previous layer and add the node bias
                layer_output.append(
                    self._activation_function(
                        self._transfer_function(
                            previous_layer_output, node_bias, node_weights
                        ),
                        function=layer_activation,
                    )
                )
            previous_layer_output = np.array(layer_output)
            layer_outputs.append(previous_layer_output)
        return layer_outputs

    def train(self, X: List[List[float]], y: List[List[float]], epochs: int) -> None:
        """Update the values of the weights and biases of the network
        based on the training data using gradient descent and mse as error function
        
        Arguments:
            X {List[List[float]]} -- training data
            y {List[List[float]]} -- target data
            epochs {int} -- amount of steps to train on
        """

        assert len(X) == len(y), "training data and target data must have equal length"

        costs = []
        for _ in range(epochs):
            final_network_dw = []
            final_network_db = []
            avg_cost = []
            for row, target in zip(X, y):
                assert len(row) == self.topology[0], "input data dimensions must match network input amount"
                assert len(target) == self.topology[-1], "target data dimensions must match network input amount"

                layer_outputs = self._get_layer_outputs(row)
                cost = pow(layer_outputs[-1] - target, 2)
                avg_cost.append(cost)
                
                layer_error = 2 * (layer_outputs[-1] - target)

                network_dw = []
                network_db = []

                # get network dw and db
                for reverse_layer_index, layer_output in enumerate(
                    reversed(layer_outputs)
                ):
                    layer_index = len(self.topology) - 1 - reverse_layer_index

                    # index of the weights and biases is offset by 1 since the input layer is skipped
                    layer_wb_index = layer_index - 1
                    layer_activation = self.activations[layer_wb_index]
                    layer_dw = []
                    layer_db = []

                    # get layer dw and db
                    for node_index, (node_output, node_error) in enumerate(
                        zip(layer_output, layer_error)
                    ):
                        node_weights = self.weights[layer_wb_index][node_index]
                        node_bias = self.biases[layer_wb_index][node_index]
                        previous_layer_output = layer_outputs[layer_wb_index - 1]
                        if layer_wb_index == 0:
                            previous_layer_output = row

                        node_input = self._transfer_function(
                            previous_layer_output, node_bias, node_weights
                        )

                        d_weights = (
                            node_error
                            * self._activation_function(
                                node_input, function=layer_activation, deriv=True
                            )
                            * previous_layer_output
                        )

                        layer_dw.append(d_weights)

                        d_bias = node_error * self._activation_function(
                            node_input, function=layer_activation, deriv=True
                        )
                        layer_db.append(d_bias)

                    network_dw.append(layer_dw)
                    network_db.append(layer_db)

                    previous_layer_error = []

                    # get previous layer error
                    for previous_node_index in range(self.topology[layer_index - 1]):
                        previous_node_error = 0

                        # get previous node error
                        for node_index, (node_output, node_error) in enumerate(
                            zip(layer_output, layer_error)
                        ):
                            node_weights = self.weights[layer_wb_index][node_index]
                            node_bias = self.biases[layer_wb_index][node_index]
                            previous_layer_output = layer_outputs[layer_wb_index - 1]
                            if layer_wb_index == 0:
                                previous_layer_output = row
                            connecting_weight = node_weights[previous_node_index]

                            node_input = self._transfer_function(
                                previous_layer_output, node_bias, node_weights
                            )

                            previous_node_error += (
                                node_error
                                * self._activation_function(
                                    node_input, function=layer_activation, deriv=True
                                )
                                * connecting_weight
                            )

                        previous_layer_error.append(previous_node_error)
                    layer_error = previous_layer_error

                network_dw = network_dw[::-1]
                network_db = network_db[::-1]
                final_network_dw.append(network_dw)
                final_network_db.append(network_db)

            avg_cost = sum(avg_cost) / len(avg_cost)
            costs.append(avg_cost)

            # average out all network dw and db
            avg_network_dw = final_network_dw[0]
            for i in range(len(self.weights)):
                for j in range(len(self.weights[i])):
                    for k in range(len(self.weights[i][j])):
                        changes = []
                        total_changes = len(final_network_dw)
                        for x in range(total_changes):
                            changes.append(final_network_dw[x][i][j][k])
                        avg_network_dw[i][j][k] = sum(changes) / total_changes

            avg_network_db = final_network_db[0]
            for i in range(len(self.biases)):
                for j in range(len(self.biases[i])):
                    changes = []
                    total_changes = len(final_network_db)
                    for x in range(total_changes):
                        changes.append(final_network_db[x][i][j])
                    avg_network_db[i][j] = sum(changes) / total_changes

            # update network weights and biases
            for i in range(len(self.weights)):
                for j in range(len(self.weights[i])):
                    for k in range(len(self.weights[i][j])):
                        self.weights[i][j][k] -= (
                            avg_network_dw[i][j][k] * self.learning_rate
                        )
            for i in range(len(self.weights)):
                for j in range(len(self.weights[i])):
                    self.biases[i][j] -= avg_network_db[i][j] * self.learning_rate

        plt.plot(costs)
        plt.show()


if __name__ == "__main__":
    n = NeuralNetwork([2, 10, 1], activations=["relu", "sigmoid"], learning_rate=0.5)
    xor_train = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    xor_test = np.array([[0], [1], [1], [0]])
    n.train(xor_train, xor_test, 1000)
    print(n.feed_forward(xor_train[0]))
    print(n.feed_forward(xor_train[1]))
    print(n.feed_forward(xor_train[2]))
    print(n.feed_forward(xor_train[3]))
