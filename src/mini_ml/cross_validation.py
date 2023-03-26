"""
A module containing a class for training a model with cross validation.
"""
from typing import Callable, Tuple

import numpy as np

from .models.base_model import BaseModel


class CrossValidationModel:
    """
    Train a model with cross validation and return the average score
    as well as a model trained on the entire dataset.
    """

    def __init__(self, model: BaseModel, metric: Callable) -> None:
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
        Randomly split the data into training and testing sets for cross-validation.

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray): The target values.
            fold (int): The current fold index.
            cross_val_folds (int): The number of cross-validation folds.

        Returns:
            A tuple containing the training data, training targets, testing data,
            and testing targets.
        """
        np.random.seed(0)
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        test_indices = indices[fold::cross_val_folds]
        train_indices = np.delete(indices, test_indices)
        X_train = X[train_indices]  # pylint: disable=invalid-name
        y_train = y[train_indices]
        X_test = X[test_indices]  # pylint: disable=invalid-name
        y_test = y[test_indices]
        return X_train, y_train, X_test, y_test

    def fit(
        self,
        X: np.ndarray,  # pylint: disable=invalid-name
        y: np.ndarray,  # pylint: disable=invalid-name
        cross_val_folds: int = 5,
    ) -> Tuple[float, BaseModel]:
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
        X: np.ndarray,  # pylint: disable=invalid-name
        y: np.ndarray,  # pylint: disable=invalid-name
        cross_val_folds: int,
    ) -> float:
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
        return float(scores.mean())
