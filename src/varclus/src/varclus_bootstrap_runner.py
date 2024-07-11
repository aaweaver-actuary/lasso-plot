"""Bootstrap resampling for estimating variable clustering probabilities."""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from varclus.interfaces import FeatureClusteringRunner, FeatureClusteringResampler
from varclus.logger import logger

__all__ = ["VarclusBootstrapRunner"]


@dataclass
class VarclusBootstrapRunner:
    """Represents the bootstrap resampling process itself.

    Algorithm
    ---------
    1. Draw a resampled dataset from the input data.
    2. Run the VarClus algorithm on the resampled dataset.
    3. Update the pair count with the counts from the VarClus clustering.
    4. Repeat steps 1-3 for a specified number of iterations.

    Parameters
    ----------
    data : pd.DataFrame
        The input data to be clustered.
    runner : FeatureClusteringRunner
        A runner instance that will be used to run the VarClus algorithm.
    resampler : FeatureClusteringResampler
        A resampler instance that will be used to draw resampled datasets.
    bootstrap_sample_fraction : float, default=0.75
        The fraction of the data to be used for each bootstrap sample.
    n_observations : int, default=2000
        The number of observations to be included in each bootstrap sample.
    n_iterations : int, default=100
        The number of bootstrap iterations to run.

    Returns
    -------
    A dictionary mapping pairs of features to the number of times out of the total iterations
    that the two features were included in the same cluster.
    """

    data: pd.DataFrame
    runner: FeatureClusteringRunner
    resampler: FeatureClusteringResampler
    bootstrap_sample_fraction: float = 0.75
    n_observations: int = 2000
    n_iterations: int = 100
    total_iterations: int = 0
    _pair_count: dict[tuple[int, int], int] | None = None

    def __post_init__(self):
        """Initialize the VarclusBootstrapRunner."""
        if not 0 < self.bootstrap_sample_fraction <= 1:
            raise ValueError("Initial sample fraction must be between 0 and 1")

    @property
    def num_features(self) -> int:
        """Return the number of features in the input data."""
        return len(self.data.columns)

    @property
    def pair_count(self) -> dict[tuple[int, int], int]:
        """Return the pair count dictionary."""
        return self._pair_count if self._pair_count is not None else {}

    @pair_count.setter
    def pair_count(self, value: dict[tuple[int, int], int]) -> None:
        """Set the pair count dictionary."""
        self._pair_count = value

    def resampled_data_draw(self) -> FeatureClusteringRunner:
        """Run the resample method on the runner to generate a new runner instance.

        Note
        ----
        This is Step 1 from the algorithm above.
        """
        # logger.debug("Starting VarclusBootstrapRunner.resampled_data_draw")
        # logger.debug("===================================================\n")

        if not 0 < self.bootstrap_sample_fraction <= 1:
            raise ValueError("Initial sample fraction must be between 0 and 1")

        return self.resampler.resample(
            fraction=self.bootstrap_sample_fraction,
            n_observations=self.n_observations,
            **self.runner.params,
        )

    def run_varclus_on_runner(self, runner: FeatureClusteringRunner) -> np.ndarray:
        """Run the VarClus algorithm on the resampled dataset.

        Note
        ----
        This is Step 2 from the algorithm above.

        Parameters
        ----------
        runner : FeatureClusteringRunner
            A runner instance that has been resampled.

        Returns
        -------
        An array of cluster assignments.
        """
        # logger.debug("Starting VarclusBootstrapRunner.run_varclus_on_runner")
        # logger.debug("=====================================================\n")

        return runner.run()

    def update_pair_count(
        self, pair_count: dict, clusters: np.ndarray
    ) -> dict[tuple[int, int], int]:
        """Update the pair_count dictionary by adding relationships from the input cluster assignments.

        If two indices i < j are included in the same cluster group G:
            pair_count[(i, j)] += 1

        For j > i, there is no entry pair_count[(j, i)], because the relationship is symmetric between
        i and j as well as between j and i.

        Note
        ----
        This is Step 3 from the algorithm above.

        Parameters
        ----------
        pair_count : dict[tuple[int, int], int]
            A dictionary mapping a pair of integers to the count representing the number of times
            that pair has been seen inside a single cluster.
        clusters: np.ndarray
            An array of cluster labels with one element for each feature being clustered.

        Returns
        -------
        dict[tuple[int, int], int]
            The updated pair_count dictionary.
        """
        # logger.debug("Starting VarclusBootstrapRunner.update_pair_count")
        # logger.debug("=================================================\n")

        # pair_count = self.pair_count.copy()
        for cluster in np.unique(clusters):
            logger.debug(f"Cluster: {cluster}")
            cluster_indices = np.where(clusters == cluster)[0]
            logger.debug(f"Cluster indices:\n\n{cluster_indices}")
            for i, j in zip(*np.triu_indices(len(cluster_indices), k=1)):
                if (cluster_indices[i], cluster_indices[j]) in pair_count:
                    pair_count[(cluster_indices[i], cluster_indices[j])] += 1
                else:
                    pair_count[(cluster_indices[i], cluster_indices[j])] = 1

        return pair_count

    def bootstrap_loop(
        self, pair_count: dict[tuple[int, int], int]
    ) -> dict[tuple[int, int], int]:
        """Bootstrap iteration loop.

        Updates the pair_count dictionary in place with the results of each iteration.

        Note
        ----
        This is a more efficient version of the algorithm that combines the resampling,
        clustering, and updating steps into a single method. It is essentiall Step 1-3
        from the algorithm above.

        Parameters
        ----------
        pair_count : dict[tuple[int, int], int]
            A dictionary mapping a pair of integers to the count representing the number of times
            that pair has been seen inside a single cluster.

        Returns
        -------
        dict[tuple[int, int], int]
            The updated pair_count dictionary.
        """
        # logger.debug("Starting VarclusBootstrapRunner.bootstrap_loop")
        # logger.debug("==============================================\n")

        for _ in range(self.n_iterations):
            # Step 1: Draw a resampled dataset
            runner = self.resampled_data_draw()

            # Step 2: Run the VarClus algorithm on the resampled dataset
            clusters = self.run_varclus_on_runner(runner)

            # Step 3: Update the pair count with the counts from the VarClus clustering
            # and increment the total number of iterations
            logger.debug(f"Current pair_count length: {len(pair_count)}")
            pair_count = self.update_pair_count(pair_count, clusters)
            self.total_iterations += 1

        # logger.debug(f"Pair count: {pair_count}")

        return pair_count
