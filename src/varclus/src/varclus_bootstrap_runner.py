"""Bootstrap resampling for estimating variable clustering probabilities."""

from __future__ import annotations
import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass
from varclus.logger import logger
from varclus.interfaces import FeatureClusteringRunner

__all__ = ["VarclusBootstrapRunner"]


@dataclass
class VarclusBootstrapRunner:
    """A runner for the VarClus algorithm that supports bootstrapping."""

    runner: FeatureClusteringRunner
    data: pd.DataFrame
    initial_sample_fraction: float = 0.75
    n_observations: int = 2000
    n_iterations: int = 100
    min_clusters: int = 2
    max_clusters: int = 20
    total_iterations: int = 0

    def __post_init__(self):
        """Initialize the VarclusBootstrapRunner."""
        self.pair_count = defaultdict(int)
        self.pairing_probabilities = defaultdict(float)
        self.num_features = len(self.data.columns.tolist())

    def resampled_data_draw(self) -> pd.DataFrame:
        """Draw a resampled dataset from the input data."""
        if not 0 < self.initial_sample_fraction <= 1:
            raise ValueError("Initial sample fraction must be between 0 and 1")

        if self.initial_sample_fraction < 1:
            data = self.data.sample(
                frac=self.initial_sample_fraction, replace=False, axis=0
            )

        return data.sample(n=self.n_observations, replace=True, axis=0)

    def calculate_pairing_probabilities(self) -> dict:
        """Calculate pairing probabilities from the pair count."""
        logger.debug(
            f"| `calculate_pairing_probabilities` | len(self.pair_count): {len(self.pair_count)}"
        )
        return {
            pair: count / self.total_iterations
            for pair, count in self.pair_count.items()
        }

    def update_pair_count(self, clusters: np.ndarray, pair_count: dict) -> dict:
        """Update the pair count based on the current clustering."""
        logger.debug(
            f"| `update_pair_count` | Current pair count length: {len(pair_count)}"
        )
        logger.debug(f"| `update_pair_count` | clusters: {clusters}")
        num_features = len(clusters)
        logger.debug(f"| `update_pair_count` | Number of features: {num_features}")
        for i in range(num_features):
            for j in range(i, num_features):
                if (i == j) or (clusters[i] == clusters[j]):
                    pair_count[(i, j)] += 1

        return pair_count

    def single_bootstrap_iteration(self, n_clusters: int, pair_count: dict) -> None:
        """Run a single bootstrap iteration."""
        runner = self.runner.resample(
            self.runner, self.initial_sample_fraction, self.n_observations
        )

        logger.debug(
            f"| `single_bootstrap_iteration` | type(runner)/runner: {type(runner)}/{runner}"
        )

        clusters = runner.run(n_clusters)

        logger.debug(
            f"| `single_bootstrap_iteration` | type(clusters)/clusters: {type(clusters)}/{clusters}"
        )
        pair_count = self.update_pair_count(clusters, pair_count)

        logger.debug(
            f"| `single_bootstrap_iteration` | type(pair_count)/len(pair_count): {type(pair_count)}/{len(pair_count)}"
        )
        self.total_iterations += 1
        return pair_count

    def bootstrap_procedure_for_one_set_of_settings(self, n_clusters: int) -> dict:
        """Run the bootstrap procedure for a single set of varclus settings."""
        temp_pair_count = defaultdict(int)
        for _ in range(self.n_iterations):
            temp_pair_count = self.single_bootstrap_iteration(
                n_clusters, temp_pair_count
            )
            if (_ + 1) % 20 == 0:
                logger.info(f"Bootstrap iteration {_ + 1}/{self.n_iterations}")
                logger.debug(
                    f"| `bootstrap_procedure_for_one_set_of_settings` | len(temp_pair_count): {len(temp_pair_count)}"
                )

        return self.pair_count

    def full_bootstrap_procedure(self) -> dict:
        """Run the full bootstrap procedure."""
        full_result = defaultdict(int)
        for n_clusters in range(self.min_clusters, self.max_clusters + 1):
            logger.info(
                f"\n\nRunning bootstrap for {n_clusters} clusters:\n=============================================="
            )
            full_result[n_clusters] = self.bootstrap_procedure_for_one_set_of_settings(
                n_clusters
            )

        logger.debug(
            f"| `full_bootstrap_procedure` | type(full_result): {type(full_result)}"
        )
        logger.debug(
            f"| `full_bootstrap_procedure` | len(full_result): {len(full_result)}"
        )
        logger.debug(
            f"| `full_bootstrap_procedure` | first 5 keys from full_result: {list(full_result.keys())[:5]}"
        )
        logger.debug(
            f"| `full_bootstrap_procedure` | first 5 values from full_result: {list(full_result.values())[:5]}"
        )

        # Combine the dicts across all cluster numbers
        output = defaultdict(float)
        for count_dict in full_result.values():
            for pair, count in count_dict.items():
                if pair in output:
                    output[pair] += count
                else:
                    output[pair] = count

        logger.debug(f"| `full_bootstrap_procedure` | type(output): {type(output)}")
        logger.debug(f"| `full_bootstrap_procedure` | len(output): {len(output)}")
        logger.debug(
            f"| `full_bootstrap_procedure` | first 5 keys from output: {list(output.keys())[:5]}"
        )
        logger.debug(
            f"| `full_bootstrap_procedure` | first 5 values from output: {list(output.values())[:5]}"
        )

        return output

    def build_probability_matrix(self) -> pd.DataFrame:
        """Build a DataFrame from the pairing probabilities."""
        probability_matrix = np.zeros((self.num_features, self.num_features))

        # DEBUGGING STATEMENTS
        logger.debug(
            f"| `build_probability_matrix` | Number of features: {self.num_features}"
        )
        logger.debug(
            f"| `build_probability_matrix` | Probability matrix shape: {probability_matrix.shape}"
        )

        # Fill in the probability matrix
        for (i, j), prob in self.pairing_probabilities.items():
            probability_matrix[i, j] = prob
            probability_matrix[j, i] = prob  # since the graph is undirected

            # DEBUGGING STATEMENTS
            logger.debug(
                f"| `build_probability_matrix` | Pair ({i}, {j}) has probability {prob:.1%}"
            )
            logger.debug(
                f"| `build_probability_matrix` | Matrix value at ({i}, {j}): {probability_matrix[i, j]:.1%}"
            )

        # Return the probability matrix as a DataFrame
        return pd.DataFrame(
            probability_matrix, columns=self.data.columns, index=self.data.columns
        )

    def run_bootstrap(self) -> pd.DataFrame:
        """Run bootstrap resampling and clustering to estimate variable clustering probabilities."""
        pair_count = self.full_bootstrap_procedure()

        logger.debug(f"| `run_bootstrap` | type(pair_count): {type(pair_count)}")
        logger.debug(f"| `run_bootstrap` | len(pair_count): {len(pair_count)}")

        # Convert to regular dict
        pair_count = dict(pair_count)

        logger.debug(
            f"| `run_bootstrap` | Dimensions of the full result: {len(pair_count)}"
        )
        pairing_probabilities = {
            n_clusters: self.calculate_pairing_probabilities()
            for n_clusters, counts in pair_count.items()
        }

        # Combining results across all cluster numbers
        combined_probabilities = defaultdict(float)
        for cluster_probabilities in pairing_probabilities.values():
            for pair, prob in cluster_probabilities.items():
                combined_probabilities[pair] += prob / (
                    self.max_clusters - self.min_clusters + 1
                )

        output = self.build_probability_matrix()

        logger.debug(f"| `run_bootstrap` | type(output): {type(output)}")
        logger.debug(f"| `run_bootstrap` | output shape: {output.shape}")
        logger.debug(f"| `run_bootstrap` | output columns: {output.head()}")

        output.to_parquet("pairing_probabilities.parquet")

        return output
