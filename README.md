# Spinning Up Tom (Thats me!)

The point of this project is to learn everything that the
SpinningUp website has to offer. The end product should be a
modular machine learning library that can easily be used to
implement anything from a basic ANN to advanced algorithms
like PPO and SAC, in a reasonably efficient way. The idea is
to implement everything from the ground up, but once something
is working, treat it as "black box" and use the most efficient
module to implement it (i.e. implement neural networks with
numpy and then with Tensorflow when completed)

## Roadmap

1. ANN module using only numpy that supports:

   1. Forwards & backwards propagation - Done!
   2. Any activation function for any layer - Done!
   3. Dynamic layer building (like Keras) - Done!
   4. BONUS: Convolutional layers

2. ANN module using tensorflow - Done!

   1. Implement training over more than 1 sample   
   2. BONUS: Implement dropout
   3. BONUS: Implement another optimizer other than SGD

3. Implement Vanilla Policy Gradient, with a solid understanding
   of the match behind it
