import tensorflow as tf


class NeuralNetwork:
    def __init__(self, layers, loss_function=tf.nn.softmax_cross_entropy_with_logits):
        self.layers = []
        first_layer = layers
        while first_layer.previous_layer:
            self.layers.insert(0, first_layer)
            first_layer = first_layer.previous_layer
        self.layers.insert(0, first_layer)
        self.last_layer = self.layers[-1]
        self.loss_function = loss_function

    def __call__(self, inputs):
        return self.last_layer.predict(inputs)

    def train_epoch(self, x_train, y_train, learning_rate=0.01, return_loss=False):
        losses = []
        weight_gradients = [tf.zeros(shape=layer.weights.shape) for layer in self.layers]
        bias_gradients = [tf.zeros(shape=layer.biases.shape) for layer in self.layers]
        for x, y_true in zip(x_train, y_train):
            with tf.GradientTape(persistent=True) as t:
                for layer in self.layers:
                    t.watch(layer.weights)
                    t.watch(layer.biases)
                y_pred = self(x)
                loss = self.loss_function(y_true, y_pred)
                losses.append(loss)

            for i, layer in enumerate(self.layers):
                weight_gradients[i] += t.gradient(loss, layer.weights) / len(x)
                bias_gradients[i] += t.gradient(loss, layer.biases) / len(x)
        
        for i, layer in enumerate(self.layers):
            layer.weights.assign_sub(weight_gradients[i] * learning_rate)
            layer.biases.assign_sub(bias_gradients[i] * learning_rate)

        if return_loss:
            return tf.reduce_mean(loss)

    def train(self, x, y, epochs, learning_rate=0.001, verbose=False):
        for epoch in range(epochs):
            loss = self.train_epoch(x, y, learning_rate, return_loss=True)
            if verbose:
                if epoch % 1 == 0:
                    print(f">epoch {epoch}: loss={loss}")


if __name__ == "__main__":
    from dense import Dense

    d = Dense(3, input_dim=2, activation="sigmoid")
    d = Dense(2, activation="softmax", previous_layer=d)
    n = NeuralNetwork(d, loss_function=tf.keras.losses.binary_crossentropy)
    x = [
        tf.constant([[1], [1]], dtype=tf.float32),
        tf.constant([[1], [0]], dtype=tf.float32),
        tf.constant([[0], [1]], dtype=tf.float32),
        tf.constant([[0], [0]], dtype=tf.float32),
    ]
    y = [
        tf.constant([[1, 0]], dtype=tf.float32),
        tf.constant([[0, 1]], dtype=tf.float32),
        tf.constant([[0, 1]], dtype=tf.float32),
        tf.constant([[1, 0]], dtype=tf.float32),
    ]
    n.train(x, y, epochs=1000, learning_rate=0.01, verbose=True)
