import tensorflow as tf
from neural_network import NeuralNetwork
from dense import Dense

print(f"Executing Eagerly: {tf.executing_eagerly()}")
print(tf.__version__)

d = Dense(3, input_dim=2)
d = Dense(1, activation="sigmoid", previous_layer=d)
n = NeuralNetwork(d)

x = tf.ones(shape=(2, 1))
y = tf.zeros(shape=(1, 1))

print(n(x))
n.train([x], [y], 100, learning_rate=10)
print(n(x))

