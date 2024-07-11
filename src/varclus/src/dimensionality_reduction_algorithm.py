"""Contains the dimensionality reduction algorithm interface and the PCA algorithm implementation."""

from typing import Protocol
import numpy as np
from sklearn.decomposition import PCA as SKLEARN_PCA
from dataclasses import dataclass

__all__ = ["DimensionalityReductionAlgorithm", "SkPCA"]


class DimensionalityReductionAlgorithm(Protocol):
    """Represents a dimensionality reduction algorithm interface."""

    def fit(self, data: np.ndarray) -> None:
        """Fit the algorithm to the data."""
        ...

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform the input data using the fitted algorithm."""
        ...

    @property
    def most_important_component(self) -> np.ndarray:
        """Return the most important component as identified by the algorithm."""
        ...

    @property
    def second_most_important_component(self) -> np.ndarray:
        """Return the second most important component as identified by the algorithm."""
        ...


@dataclass
class SkPCA(DimensionalityReductionAlgorithm):
    """A dimensionality reduction algorithm based on the PCA algorithm, scikit-learn implementation."""

    model: SKLEARN_PCA
    n_components: int
    random_state: int = 42

    def __post_init__(self):
        """Initialize the PCA algorithm."""
        self.model = SKLEARN_PCA(
            n_components=self.n_components, random_state=self.random_state
        )

    def fit(self, data: np.ndarray) -> None:
        """Fit the PCA algorithm to the data and return the transformed data.

        Parameters
        ----------
        data : np.ndarray
            The data to fit the PCA algorithm to and transform.

        Returns
        -------
        np.ndarray
            The PCA-transformed data.
        """
        return self.model.fit_transform(data)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform the data using the fitted PCA algorithm.

        Parameters
        ----------
        data : np.ndarray
            The data to transform.

        Returns
        -------
        np.ndarray
            The transformed data.
        """
        return self.model.transform(data)

    @property
    def most_important_component(self) -> np.ndarray:
        """Return the most important principal component."""
        return self.model.components_[0]

    @property
    def second_most_important_component(self) -> np.ndarray:
        """Return the second most important principal component."""
        return self.model.components_[1]
