"""Interfaces for feature clustering algorithms."""

import numpy as np
from typing import Protocol
from dataclasses import dataclass


@dataclass
class FeatureClusteringRunner(Protocol):
    """A protocol for feature clustering algorithms. At minimum, the algorithm must implement a `run` method that returns an array of cluster assignments, and a `resample` method that returns a new instance of the algorithm with resampled data."""

    def run(self) -> np.ndarray: ...  # noqa: D102

    @classmethod
    def resample(  # noqa: D102
        cls, instance: "FeatureClusteringRunner", fraction: float, n_observations: int
    ) -> "FeatureClusteringRunner": ...


@dataclass
class FeatureClusteringBootstrapRunner(Protocol):
    """A protocol for feature clustering algorithms that support bootstrapping. At minimum, the algorithm must implement a `run_bootstrap` method that returns a matrix of pairing probabilities."""

    def run_bootstrap(self) -> np.ndarray: ...  # noqa: D102


@dataclass
class FeatureClusterer(Protocol):
    """A protocol for feature clustering algorithms. At minimum, the algorithm must implement a `cluster` method that returns a dictionary mapping feature indices to cluster labels."""

    def cluster(self) -> dict: ...  # noqa: D102
