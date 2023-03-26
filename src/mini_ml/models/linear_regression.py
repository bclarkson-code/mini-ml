"""
An implementation of linear regression using numpy.
"""
import numpy as np

from .base_model import BaseModel


class LinearRegression(BaseModel):
    """
    A linear regression model implemented with pure numpy.
    """

    coef_: np.ndarray = np.array([])

    def __init__(self) -> None:
        """
        Initialize LinearRegression object.
        """

    def _prepare_data(
        self, X: np.ndarray  # pylint: disable=invalid-name
    ) -> np.ndarray:
        """
        Prepare data for fitting by adding a column of ones to the
        front of the matrix.

        Parameters
        ----------
        X : numpy.ndarray
            Input features.

        Returns
        -------
        numpy.ndarray
            Prepared data.
        """
        ones = np.ones((X.shape[0], 1))
        return np.column_stack((ones, X))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
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
        ones = np.ones((X.shape[0], 1))
        X = np.column_stack((ones, X))
        self.coef_ = np.linalg.inv(X.T @ X) @ X.T @ y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
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
        intercept = self.coef_[0]
        return X @ self.coef_[1:] + intercept

    def get_params(self):
        """
        Get model parameters.
        """
        return {"coef_": self.coef_}

    def set_params(self, params: dict) -> "LinearRegression":
        """
        Load model parameters.

        Parameters
        ----------
        params : dict
            Model parameters.

        Returns
        -------
        LinearRegression
            Model with loaded parameters.
        """
        self.coef_ = params["coef_"]
        return self
