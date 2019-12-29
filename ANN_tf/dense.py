import tensorflow as tf

ACTIVATIONS = {
    "sigmoid": tf.math.sigmoid,
    "relu": tf.nn.relu,
    "softmax": tf.nn.softmax,
    None: None,
}


def generate_weights_and_biases(layer_size, previous_size):
    initializer = tf.initializers.RandomNormal()
    config = initializer.get_config()
    initializer = tf.initializers.RandomNormal.from_config(config)
    weights = initializer(shape=(layer_size, previous_size))
    biases = initializer(shape=(layer_size, 1))
    return tf.Variable(weights), tf.Variable(biases)


class Dense:

    def __init__(self, size: int, input_dim: int = None, activation: str = None,
                 previous_layer=None):
        self.size = size
        self.activation = ACTIVATIONS[activation]
        self.input_dim = input_dim
        self.previous_layer = previous_layer
        self.layer_activation = tf.zeros(shape=(self.size, 1), dtype=tf.float32)
        self.weights = None
        self.biases = None

        if previous_layer:
            self.previous_layer = previous_layer
            self.input_dim = previous_layer.size

        if self.input_dim:
            self.initialize_weights_and_biases()

    def __call__(self, previous_layer: object):
        self.previous_layer = previous_layer
        self.input_dim = previous_layer.size
        self.initialize_weights_and_biases()
        return self

    def initialize_weights_and_biases(self):
        self.weights, self.biases = generate_weights_and_biases(self.size,
                                                                self.input_dim)

    @tf.function()
    def predict(self, inputs):
        """Receives a (self.input_dim, 1) sized tensor
        """
        if self.previous_layer:
            layer_input = self.previous_layer.predict(inputs)
        else:
            layer_input = inputs

        layer_transfer = tf.add(tf.matmul(self.weights, layer_input),
                                self.biases)
        if self.activation:
            if self.activation is ACTIVATIONS["softmax"]:
                self.layer_activation = self.activation(layer_transfer, axis=0)
            else:
                self.layer_activation = self.activation(layer_transfer)
        else:
            self.layer_activation = layer_transfer
        return self.layer_activation
