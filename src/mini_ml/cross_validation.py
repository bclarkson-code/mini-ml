"""
A module containing a class for training a model with cross validation.
"""
from typing import Tuple

import numpy as np

from .models.base_model import BaseModel


class CrossValidationModel:
    """
    Train a model with cross validation and return the average score
    as well as a model trained on the entire dataset.
    """

    def __init__(self, model: object, metric: callable) -> None:
        """
        Initialize CrossValidationModel object.

        Parameters
        ----------
        model : object
            Model to be trained with cross validation.
        metric : callable
            Scoring function to evaluate model performance.

        Returns
        -------
        None
        """
        self.model = model
        self.metric = metric

    def _split(
        self,
        X: np.ndarray,  # pylint: disable=invalid-name
        y: np.ndarray,  # pylint: disable=invalid-name
        fold: int,
        cross_val_folds: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split the data into training and testing sets for cross-validation.

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray): The target values.
            fold (int): The current fold index.
            cross_val_folds (int): The number of cross-validation folds.

        Returns:
            A tuple containing the training data, training targets, testing data,
            and testing targets.
        """
        len_data = X.shape[0]
        fold_size = len_data // cross_val_folds
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size
        X_train = np.vstack(  # pylint: disable=invalid-name
            [X[:test_start], X[test_end:]]
        )
        y_train = np.hstack([y[:test_start], y[test_end:]])
        X_test = X[test_start:test_end]  # pylint: disable=invalid-name
        y_test = y[test_start:test_end]
        return X_train, y_train, X_test, y_test

    def fit(
        self,
        X: np.ndarray,  # pylint: disable=invalid-name
        y: np.ndarray,  # pylint: disable=invalid-name
        cross_val_folds: int = 5,
    ) -> Tuple[np.float64, BaseModel]:
        """
        Fit model to input data with cross validation.

        Parameters
        ----------
        X : numpy.ndarray
            Input features.
        y : numpy.ndarray
            Target values.
        cross_val_folds : int, optional
            Number of cross validation folds, by default 5.

        Returns
        -------
        tuple
            Tuple containing the average score and a model trained on the entire dataset.
        """
        score = self.score(X, y, cross_val_folds=cross_val_folds)
        model = self.model.fit(X, y)
        return score, model

    def score(
        self,
        X,  # pylint: disable=invalid-name
        y,  # pylint: disable=invalid-name
        cross_val_folds,
    ):
        """
        Score the performance of the model on the input data with cross validation.
        """
        scores = np.zeros(cross_val_folds)
        for fold in range(cross_val_folds):
            (
                X_train,  # pylint: disable=invalid-name
                y_train,
                X_test,  # pylint: disable=invalid-name
                y_test,
            ) = self._split(X, y, fold, cross_val_folds)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            score = self.metric(y_test, y_pred)
            scores[fold] = score
        return scores.mean()
