import tensorflow as tf


class NeuralNetwork:

    def __init__(self, layers):
        self.layers = []
        first_layer = layers
        while first_layer.previous_layer:
            self.layers.insert(0, first_layer)
            first_layer = first_layer.previous_layer
        self.layers.insert(0, first_layer)
        self.last_layer = self.layers[-1]

    def __call__(self, inputs):
        return self.last_layer.predict(inputs)

    def train_epoch(self, x, y_true, learning_rate=0.01):
        with tf.GradientTape(persistent=True) as t:
            for layer in self.layers:
                t.watch(layer.weights)
                t.watch(layer.biases)
            y_pred = self(x)
            loss = tf.metrics.mse(y_true, y_pred)

        for layer in self.layers:
            layer.weights.assign_sub(t.gradient(loss, layer.weights) * learning_rate)
            layer.biases.assign_sub(t.gradient(loss, layer.biases) * learning_rate)

    def train(self, x, y, epochs, learning_rate=0.01):
        for _ in range(epochs):
            for row, target in zip(x, y):
                self.train_epoch(row, target, learning_rate)

