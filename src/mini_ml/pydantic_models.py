"""
Pydantic models for the API
"""
from abc import ABC
from typing import Callable, List, Literal

import numpy as np
from pydantic import BaseModel as PydanticBaseModel  # pylint: disable=no-name-in-module

from .models.base_model import BaseModel
from .parse import load_data, load_metric, load_model


class TrainRequest(PydanticBaseModel):  # pylint: disable=too-few-public-methods
    """
    The raw request from the client to train a model
    """

    model: Literal["linear_regression"]
    metric: Literal["r_squared"]
    X: List[List[float]]
    y: List[float]
    name: str


class ParsedTrainRequest(ABC):  # pylint: disable=too-few-public-methods
    """
    The parsed request from the client to train a model
    """

    model: BaseModel
    metric: Callable
    X: np.ndarray
    y: np.ndarray
    name: str

    def __init__(self, train_request: TrainRequest, **data):
        super().__init__(**data)
        self.model = load_model(train_request.model)
        self.metric = load_metric(train_request.metric)
        self.X, self.y = load_data(  # pylint: disable=invalid-name
            train_request.X, train_request.y
        )
        self.name = train_request.name


class TrainResponse(PydanticBaseModel):  # pylint: disable=too-few-public-methods
    """
    The response from the server to the client after training a model
    """

    name: str
    score: float
    model: str


class PredictRequest(PydanticBaseModel):  # pylint: disable=too-few-public-methods
    """
    A request from the client to get a prediction from a model
    """

    name: str
    X: List[List[float]]
