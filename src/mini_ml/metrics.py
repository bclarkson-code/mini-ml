"""
Metrics that can be used to evaluate the performance of a model.
"""
import numpy as np


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the R squared score for a given set of true and predicted values.

    Parameters
    ----------
    y_true : numpy.ndarray
        True values.
    y_pred : numpy.ndarray
        Predicted values.

    Returns
    -------
    np.float64
    """
    y_mean = np.mean(y_true)
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - y_mean) ** 2)
    if denominator == 0 or numerator == 0:
        return 1
    return float(1 - numerator / denominator)
