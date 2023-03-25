"""
A base class for all models. All models should inherit from this class.
"""

import numpy as np


class BaseModel:
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
    ) -> None:
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
        None
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
