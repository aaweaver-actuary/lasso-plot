"""Define the interfaces and types for the project."""

from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class AbstractModel(ABC):
    """Abstract base class for models."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> None:
        """Fit the model."""

    @abstractmethod
    def get_path(self) -> tuple:
        """Get the alphas and coefficients path."""


class AbstractPlotter(ABC):
    """Abstract base class for plotting the LASSO path."""

    @abstractmethod
    def plot(self, alphas: list[float], coefs: list[float]) -> None:
        """Plot the LASSO path for a logistic regression model."""


class AbstractPreprocessor(ABC):
    """Abstract base class for data preprocessing."""

    @abstractmethod
    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data in the input feature matrix."""


class AbstractPath(ABC):
    """Abstract base class representing a single LASSO path."""

    @property
    @abstractmethod
    def alphas(self) -> list[float]:
        """Get the alphas along the regularization path."""

    @property
    @abstractmethod
    def coefs(self) -> list[float]:
        """Get the coefficients along the regularization path."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the path."""

    @abstractmethod
    def get_path(self) -> tuple:
        """Get the alphas and coefficients for a single LASSO path."""
