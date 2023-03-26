"""
A class that stores models and allows their retrieval.
"""
from abc import ABC
from pathlib import Path

import joblib  # type: ignore
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from .models.base_model import BaseModel

Base = declarative_base()


class ModelExistsError(Exception):
    """
    An exception that is raised when a model with the same name already
    exists in the store.
    """


class Model(Base):  # type: ignore # pylint: disable=too-few-public-methods
    """
    A class that represents a model in the database.
    """

    __tablename__ = "models"

    id = Column(Integer, primary_key=True)
    username = Column(String(32))
    name = Column(String(128))
    type = Column(String(128))
    path = Column(String(1024))

    def __repr__(self):
        return f"Model({self.username=}, {self.name=}, " f"{self.type=}, {self.path=})"


class ModelStore(ABC):
    """
    A base class for storing model parameters and allowing their retrieval.
    """

    def save(self, username: str, model_name: str, model: BaseModel) -> None:
        """
        Add a model to the store.

        Parameters
        ----------
        model : Model
            Model to be added to the store.

        Returns
        -------
        None
        """
        raise NotImplementedError

    def load(self, username: str, model_name: str) -> BaseModel:
        """
        Get a model from the store.

        Parameters
        ----------
        username : str
            Username of the user who trained the model.
        model_name : str
            Name of the model.

        Returns
        -------
        Model
            Model from the store.
        """
        raise NotImplementedError


class LocalModelStore(ModelStore):
    """
    A model store that stores models locally. It uses a SQLite database to
    store the locations of model binaries and stores the model binaries in
    the local file system.
    """

    def __init__(self, db_path: str, model_dir: str):
        """
        Initialize LocalModelStore object.

        Parameters
        ----------
        db_path : str
            Path to the SQLite database.

        model_dir : str
            Path to the directory where model binaries are stored.
        """
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)
        self.session = sessionmaker(bind=self.engine)()
        self.model_dir = model_dir

    def _store_model_params(
        self, username: str, model_name: str, model_params: dict
    ) -> str:
        """
        Store model parameters in a file.

        Parameters
        ----------
        username : str
            Username of the user who trained the model.
        model_name : str
            Name of the model.
        model_params : dict
            Model parameters.

        Returns
        -------
        model_path : str
            Path to the model file.
        """
        filename = f"{model_name}.joblib"
        save_dir = Path(self.model_dir) / username
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / filename

        with open(model_path, "wb") as file_stream:
            joblib.dump(model_params, file_stream)

        return str(model_path)

    def save(self, username: str, model_name: str, model: BaseModel):
        """
        Add a model to the store.

        Parameters
        ----------
        model : Model
            Model to be added to the store.

        Returns
        -------
        None
        """
        # check if the model already exists
        if (
            self.session.query(Model)
            .filter_by(username=username, name=model_name)
            .count()
        ):
            raise ModelExistsError

        # save the model
        model_params = model.get_params()
        model_path = self._store_model_params(username, model_name, model_params)

        # add the model to the database
        model = Model(
            username=username,
            name=model_name,
            path=model_path,
            type=model.__class__.__name__,
        )
        self.session.add(model)
        self.session.commit()

    def load(self, username: str, model_name: str) -> BaseModel:
        """
        Get a model from the store.

        Parameters
        ----------
        username : str
            Username of the user who trained the model.
        model_name : str
            Name of the model.

        Returns
        -------
        model : BaseModel
            Model from the store.
        """
        # Fetch the model path from the database
        model_metadata = (
            self.session.query(Model)
            .filter_by(username=username, name=model_name)
            .one()
        )

        # Load the model parameters from disk
        with open(model_metadata.path, "rb") as file_stream:
            model_params = joblib.load(file_stream)

        if model_metadata.type == "LinearRegression":
            from .models.linear_regression import (
                LinearRegression,
            )  # pylint: disable=import-outside-toplevel

            model = LinearRegression().set_params(model_params)
        return model
