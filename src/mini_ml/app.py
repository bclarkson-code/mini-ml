"""
A server that allows users to train models and get predictions from them.
"""
from typing import Tuple

import numpy as np
from fastapi import FastAPI

from .cross_validation import CrossValidationModel
from .model_store import LocalModelStore
from .models.base_model import BaseModel
from .pydantic_models import (
    ParsedTrainRequest,
    PredictRequest,
    TrainRequest,
    TrainResponse,
)

app = FastAPI()
model_store = LocalModelStore("models.db", "models")


def _train(
    train_request: ParsedTrainRequest,
) -> Tuple[float, BaseModel]:
    """
    Train a model and return the cross validation score and the model
    """
    cross_val_model = CrossValidationModel(train_request.model, train_request.metric)
    score, model = cross_val_model.fit(train_request.X, train_request.y)
    username = "test_user"
    model_store.save(username, train_request.name, model)
    return score, model


@app.post("/train")
def train(train_request: TrainRequest) -> TrainResponse:
    """
    Given some data, train a model and
    return the cross validation score
    """
    parsed_train_request = ParsedTrainRequest(train_request)
    score, model = _train(parsed_train_request)
    return TrainResponse(
        name=parsed_train_request.name,
        score=score,
        model=type(model).__name__,
    )


@app.get("/predict")
def predict(predict_request: PredictRequest):
    """
    Given some data, get a prediction from a model
    """
    model = model_store.load("test_user", predict_request.name)
    X = np.array(predict_request.X)  # pylint: disable=invalid-name
    prediction = model.predict(X).tolist()
    response = {"prediction": prediction, "model": model.__class__.__name__}
    print(response)
    return response
