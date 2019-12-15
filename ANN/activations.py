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


def linear(x: float, deriv: bool = False) -> float:
    if deriv:
        return 1
    return x


def tanh(x: float, deriv: bool = False) -> float:
    if deriv:
        return 1 - pow(tanh(x, deriv=False), 2)
    return 2.0 / (1.0 + np.exp(-2 * x))


def lrelu(x: float, deriv: bool = False) -> float:
    if deriv:
        return 0.01 if x < 0 else 1
    return 0.01 * x if x < 0 else x


ACTIVATIONS = {
    "sigmoid": sigmoid,
    "relu": relu,
    "linear": linear,
    "tanh": tanh,
    "lrelu": lrelu,
}
