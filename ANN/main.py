import numpy as np
from neural_network import NeuralNetwork
from dense_layer import Dense

# load MNIST dataset
import tensorflow as tf

DATA_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"

path = tf.keras.utils.get_file("mnist.npz", DATA_URL)
with np.load(path) as data:
    train_examples = data["x_train"]
    train_labels = data["y_train"]
    test_examples = data["x_test"]
    test_labels = data["y_test"]

# flatten input data
flat_train_examples = []
for row in train_examples:
    flat_train_examples.append(np.ndarray.flatten(row))
flat_test_examples = []
for row in test_examples:
    flat_test_examples.append(np.ndarray.flatten(row))

# train a network with two hidden layers with 10 neurons each
np.random.RandomState(seed=1)
n = NeuralNetwork(cost_function="mse", learning_rate=0.01)
n.append_layer(Dense(size=10, input_dimensions=28 * 28, activation="lrelu"))
n.append_layer(Dense(size=10, input_dimensions=10, activation="lrelu"))
n.append_layer(Dense(size=10, input_dimensions=10, activation="sigmoid"))

n.train(flat_train_examples, train_labels, 20000, verbose=True)

# print results
losses = []
guesses = []
for row, target in zip(flat_test_examples, test_labels):
    prediction = n.predict(row)
    losses.append(n.cost_function(prediction, target))
    guesses.append(np.argmax(prediction) == np.argmax(target))
print(
    f"Average loss: {np.mean(losses)}"
    f"Predictions: {len(guesses)}"
    f"Accuracy: {sum(guesses) / len(guesses) * 100}/100"
)
