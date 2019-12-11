import numpy as np
from typing import List, Union


class NeuralNetwork:
    def __init__(
        self,
        topology: List[int],
        weight_range: float = 1.0,
        bias_range: float = 1.0,
        activations: Union[str, List[str]] = "relu",
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
                return 0 if x <= 0 else 1
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
        for layer_biases, layer_weights, activation_function in zip(self.biases, self.weights, self.activations):
            layer_output = []
            for node_bias, node_weights in zip(layer_biases, layer_weights):

                # apply the activation function on the output of the previous layer and add the node bias
                layer_output.append(
                    self._activation_function(sum(previous_layer_output * node_weights), function=activation_function)
                    + node_bias
                )
            previous_layer_output = np.array(layer_output)
        return previous_layer_output


if __name__ == "__main__":
    n = NeuralNetwork([2, 3, 5], activations=["sigmoid", "relu"])
    print(n.feed_forward(np.array([1.0, 0.5])))
