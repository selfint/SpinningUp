import numpy as np
from typing import Union


def mse(
    prediction: np.array, target: np.array, deriv: bool = False
) -> Union[np.array, float]:
    if deriv:
        return 2 * (prediction - target)
    else:
        return np.mean(pow(prediction - target, 2))


def cce(
    prediction: np.array, target: np.array, deriv: bool = False
) -> Union[np.array, float]:
    if deriv:
        return -1 * (target / (prediction + 1e-15)) + (1 - target) / (1 - prediction + 1e-15)
    return -1 * sum(target * np.log(prediction))


COST_FUNCTIONS = {
    "mse": mse,
    "mean squared error": mse,
    "cce": cce,
    "catgorical cross entropy": cce,
}

