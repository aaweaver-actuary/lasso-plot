"""Interfaces for feature clustering algorithms."""

import numpy as np
from typing import Protocol
from dataclasses import dataclass


@dataclass
class FeatureClusteringRunner(Protocol):
    """A protocol for feature clustering algorithms. At minimum, the algorithm must implement a `run` method that returns an array of cluster assignments, and a `resample` method that returns a new instance of the algorithm with resampled data."""

    @property
    def params(self) -> dict:
        """Return the parameters of the feature clustering algorithm."""
        ...

    def run(self) -> np.ndarray:
        """Run the feature clustering algorithm and return an array of cluster assignments."""
        ...


@dataclass
class FeatureClusteringResampler(Protocol):
    """A protocol for feature clustering algorithms that support resampling. At minimum, the algorithm must implement a `resample` method that returns a new instance of the algorithm with resampled data."""

    def resample(
        self, fraction: float = 0.7, n_observations: int = 2000, **kwargs
    ) -> "FeatureClusteringRunner":
        """Draw a resampled dataset from the input data."""
        ...


@dataclass
class FeatureClusteringBootstrapRunner(Protocol):
    """A protocol for feature clustering algorithms that support bootstrapping. At minimum, the algorithm must implement a `run_bootstrap` method that returns a matrix of pairing probabilities."""

    def run_bootstrap(self) -> np.ndarray:
        """Run the bootstrap to calculate pairing probabilities."""
        ...


@dataclass
class FeatureClusterer(Protocol):
    """A protocol for feature clustering algorithms. At minimum, the algorithm must implement a `cluster` method that returns a dictionary mapping feature indices to cluster labels."""

    def cluster(self) -> dict:
        """Cluster features and return a dictionary mapping feature indices to cluster labels."""
        ...
