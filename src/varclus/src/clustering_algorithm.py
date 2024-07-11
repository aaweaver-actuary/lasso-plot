"""Contains the clustering algorithm interface and the Agglomerative Clustering algorithm implementation."""

from __future__ import annotations
import numpy as np
from typing import Protocol, Callable
from sklearn.cluster import FeatureAgglomeration
from dataclasses import dataclass
import joblib

__all__ = ["ClusteringAlgorithm", "SkAgglomerativeClusterer"]


class ClusteringAlgorithm(Protocol):
    """Represents a clustering algorithm interface."""

    def fit(self, data: np.ndarray) -> None:
        """Take in data and fit the clustering algorithm."""
        ...

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Take in data and predict the cluster labels."""
        ...

    @property
    def clusters(self) -> np.ndarray:
        """Return the clusters. The format of the clusters depends on the algorithm."""
        ...

    @property
    def n_clusters(self) -> int:
        """Return the number of clusters."""
        ...


@dataclass
class SkAgglomerativeClusterer(ClusteringAlgorithm):
    """A clustering algorithm based on the Agglomerative Clustering algorithm, scikit-learn implementation."""

    model: FeatureAgglomeration | None = None
    n_clusters: int = 2
    metric: str = "euclidean"  # or l1, l2, manhattan, cosine
    memory: joblib.Memory | None = None
    linkage: str = "ward"  # or complete, average, single
    pooling_func: Callable = np.mean

    def __post_init__(self):
        """Initialize the Agglomerative Clustering algorithm."""
        self.model = FeatureAgglomeration(
            n_clusters=self.n_clusters,
            metric=self.metric,
            memory=self.memory,
            linkage=self.linkage,
            pooling_func=self.pooling_func,
        )

    def fit(self, data: np.ndarray) -> None:
        """Fit the Agglomerative Clustering algorithm to the data.

        Parameters
        ----------
        data : np.ndarray
            The data to fit the clustering algorithm to.

        Returns
        -------
        None
            The method fits the algorithm in place.
        """
        self.model.fit(data)

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict the cluster labels for the data.

        Parameters
        ----------
        data : np.ndarray
            The data to predict the cluster labels for.

        Returns
        -------
        np.ndarray
            The cluster labels for the data.

        Raises
        ------
        ValueError
            If the algorithm has not been fitted already.
        """
        if not hasattr(self.model, "labels_"):
            raise ValueError("The algorithm has not been fitted yet.")
        return self.model.transform(data)

    @property
    def clusters(self) -> np.ndarray:
        """Return the clusters."""
        if not hasattr(self.model, "labels_"):
            raise ValueError("The algorithm has not been fitted yet.")
        return self.model.labels_
