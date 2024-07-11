"""Define AnClustRunner class for variable clustering."""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Callable
from dataclasses import dataclass
from joblib import Memory
from varclus.logger import logger
from varclus.src.clustering_algorithm import (
    ClusteringAlgorithm,
    SkAgglomerativeClusterer,
)


# Create a memory object to cache the results of the PCA step
memory = Memory(location="cache_dir", verbose=0)


@dataclass
class AnClustResampler:
    """Create a new instance of AnClustRunner with a bootstrapped dataset."""

    data: pd.DataFrame
    n_jobs: int = 8

    def resample(
        self, fraction: float = 0.7, n_observations: int = 2000, **kwargs
    ) -> "AnClustRunner":
        """Draw a resampled dataset from the input data."""
        if not 0 < fraction <= 1:
            raise ValueError(f"Fraction must be between 0 and 1, but got {fraction}")

        if fraction < 1:
            data = self.data.sample(frac=fraction, replace=False, axis=0)

        resampled_data = data.sample(n=n_observations, replace=True, axis=0)
        # logger.debug(f"kwargs: {kwargs}")
        return AnClustRunner(resampled_data, **kwargs)


@dataclass
class AnClustRunner:
    """Variable clustering using a slightly modified Agglomerative Clustering algorithm.

    Attributes
    ----------
    data : pd.DataFrame
        The input data for clustering.
    clustering_algo : ClusteringAlgorithm
        Clustering algorithm instance.
    n_clusters : int
        The number of clusters to create.
    metric : str, optional
        The distance metric to use for clustering (default is "euclidean").
    memory : Memory, optional
        The memory object to cache the results of the PCA step (default is None).
    linkage : str, optional
        The linkage criterion to use for clustering (default is "ward").
    pooling_func : Callable, optional
        The pooling function to use for clustering (default is np.mean).
    n_jobs : int, optional
        The number of jobs for parallel processing (default is 8).
    """

    data: pd.DataFrame
    n_clusters: int
    metric: str = "euclidean"  # or l1, l2, manhattan, cosine
    linkage: str = "ward"  # or complete, average, single
    pooling_func: Callable = np.mean
    n_jobs: int = 8
    clusterer: ClusteringAlgorithm | None = None

    @property
    def params(self) -> dict:
        """Return the parameters of the AnClustRunner."""
        return {
            "n_clusters": self.n_clusters,
            "metric": self.metric,
            "linkage": self.linkage,
            "pooling_func": self.pooling_func,
        }

    def __post_init__(self):
        """Initialize the AnClustRunner."""
        if self.linkage == "ward":
            if self.metric != "euclidean":
                logger.warning(
                    "Euclidean distance is required when using the 'ward' linkage criterion. Automatically setting metric to 'euclidean'."
                )
            self.metric = "euclidean"

        if self.clusterer is None:
            self.clusterer = SkAgglomerativeClusterer(
                n_clusters=self.n_clusters,
                metric=self.metric,
                memory=memory,
                linkage=self.linkage,
                pooling_func=self.pooling_func,
            )

    def validate_data(self) -> None:
        """Validate the input data."""
        if self.data.isna().astype(int).sum().sum() > 0:
            raise ValueError(
                "Input data contains null values. Please clean the data before running VarClus."
            )
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("Input data should be a pandas DataFrame.")

    def run(self) -> np.ndarray:
        """Run the VarClus algorithm to cluster the variables."""
        # Validate the data
        self.validate_data()

        # Set the initial cluster assignments
        correlation_matrix = self.data.corr().to_numpy()
        self.clusterer.fit(correlation_matrix)
        initial_clusters = self.clusterer.clusters
        self.clusters = initial_clusters

        return initial_clusters
