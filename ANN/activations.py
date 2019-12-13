import numpy as np


def sigmoid(x: float, deriv: bool = False) -> float:
    if deriv:
        dx = 1.0 / (1.0 + np.exp(-x))
        return dx * (1 - dx)
    return 1.0 / (1.0 + np.exp(-x))


def relu(x: float, deriv: bool = False) -> float:
    if deriv:
        return 0 if x < 0 else 1
    return max(0, x)


ACTIVATIONS = {
    "sigmoid": sigmoid,
    "relu": relu
}
