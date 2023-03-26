"""
A base class for all models. All models should inherit from this class.
"""

from abc import ABC

import numpy as np


class BaseModel(ABC):
    """
    Base class for all models.
    """

    def __init__(self) -> None:
        """
        Initialize BaseModel object.
        """
        raise NotImplementedError

    def fit(
        self,
        X: np.ndarray,  # pylint: disable=invalid-name
        y: np.ndarray,  # pylint: disable=invalid-name
    ) -> "BaseModel":
        """
        Fit linear regression model to the input data.

        Parameters
        ----------
        X : numpy.ndarray
            Input features.
        y : numpy.ndarray
            Target values.

        Returns
        -------
        BaseModel :
            Fitted model.
        """
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
        """
        Predict target values for the input data.

        Parameters
        ----------
        X : numpy.ndarray
            Input features.

        Returns
        -------
        numpy.ndarray
            Predicted target values.
        """
        raise NotImplementedError

    def get_params(self) -> dict:
        """
        Get model parameters.

        Returns
        -------
        dict
            Model parameters.
        """
        raise NotImplementedError

    def set_params(self, params: dict) -> "BaseModel":
        """
        Set model parameters.

        Parameters
        ----------
        **kwargs : dict
            Model parameters.

        Returns
        -------
        None
        """
        raise NotImplementedError
