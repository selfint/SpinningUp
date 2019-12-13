import numpy as np
from typing import Union


def mse(
    prediction: np.array, target: np.array, deriv: bool = False
) -> Union[np.array, float]:
    """Mean squared error cost function
    
    Arguments:
        prediction {np.array} -- network output
        target {np.array} -- target network output
    
    Keyword Arguments:
        deriv {bool} -- get the derivate of the cost function (default: {False})
    
    Returns:
        Union[np.array, float] -- returns the derivate of the cost function
        for each neuron in the output layer (np.array),
        or the cost of the network as a whole (float)
    """
    if deriv:
        return 2 * (prediction - target)
    else:
        return np.mean(pow(prediction - target, 2))


COST_FUNCTIONS = {"mse": mse, "mean squared error": mse}
