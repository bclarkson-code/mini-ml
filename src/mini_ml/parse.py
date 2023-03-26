"""
Parse data from a request
"""
from typing import Callable, List, Tuple

import numpy as np

from .models.base_model import BaseModel


def load_model(model_string: str) -> BaseModel:
    """
    Load a model from a string
    """
    if model_string == "linear_regression":
        from .models.linear_regression import (
            LinearRegression,
        )  # pylint: disable=import-outside-toplevel

        model = LinearRegression()
    else:
        supported_models = ["linear_regression"]
        raise NotImplementedError(
            f"Model: {model_string} not supported. Supported models: {supported_models}"
        )
    return model


def load_metric(metric_string: str) -> Callable:
    """
    Load a metric from a string
    """
    if metric_string == "r_squared":
        from .metrics import r_squared  # pylint: disable=import-outside-toplevel

        metric = r_squared
    else:
        supported_metrics = ["r_squared"]
        raise NotImplementedError(
            (
                f"Metric: {metric_string} not supported. "
                f"Supported metrics: {supported_metrics}"
            )
        )
    return metric


def load_data(
    X: List[List[float]], y: List[float]  # pylint: disable=invalid-name
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data from a request
    """
    X_array = np.array(X)  # pylint: disable=invalid-name
    y_array = np.array(y)
    return X_array, y_array
