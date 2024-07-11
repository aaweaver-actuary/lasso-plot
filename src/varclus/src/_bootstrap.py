"""Bootstrap resampling for estimating variable clustering probabilities."""

from __future__ import annotations
import numpy as np
import pandas as pd
from collections import defaultdict
from varclus.src.varclus_runner import VarclusRunner as VarClus
from varclus.logger import logger

__all__ = ["varclus_bootstrap"]


def calculate_pairing_probabilities(
    pair_count: dict[tuple[int, int], int], num_runs: int
) -> dict:
    """Calculate pairing probabilities from the pair count."""
    output = {pair: count / num_runs for pair, count in pair_count.items()}

    # DEBUGGING STATEMENTS
    logger.debug(f"| `calculate_pairing_probabilities` | pair_count: {pair_count}")
    logger.debug(f"| `calculate_pairing_probabilities` | num_runs: {num_runs}")
    logger.debug(f"| `calculate_pairing_probabilities` | output: {output}")
    return output


def resampled_data_draw(
    data: pd.DataFrame, initial_sample_fraction: float = 1.0, n_observations: int = 2000
) -> pd.DataFrame:
    """Draw a resampled dataset from the input data."""
    if not 0 < initial_sample_fraction <= 1:
        raise ValueError("Initial sample fraction must be between 0 and 1")

    if initial_sample_fraction < 1:
        data = data.sample(frac=initial_sample_fraction, replace=False, axis=0)

    return data.sample(n=n_observations, replace=True, axis=0)


def update_pair_counts(pair_count: dict, clusters: np.ndarray) -> dict:
    """Update the pair count based on the current clustering."""
    num_features = len(clusters)
    for i in range(num_features):
        for j in range(i + 1, num_features):
            if clusters[i] == clusters[j]:
                pair_count[(i, j)] += 1
    return pair_count


def single_bootstrap_iteration(
    pair_count: dict, data: pd.DataFrame, n_clusters: int
) -> dict:
    """Run a single bootstrap iteration."""
    varclus = VarClus(data=data, n_clusters=n_clusters)
    clusters = varclus.run()
    return update_pair_counts(pair_count, clusters)


def bootstrap_procedure_for_one_set_of_settings(
    data: pd.DataFrame, n_iterations: int, n_clusters: int
) -> dict:
    """Run the bootstrap procedure for a single set of varclus settings."""
    pair_count = defaultdict(int)
    for _ in range(n_iterations):
        if (_ + 1) % 20 == 0:
            logger.info(f"Bootstrap iteration {_ + 1}/{n_iterations}")
        resampled_data = resampled_data_draw(data)
        pair_count = single_bootstrap_iteration(pair_count, resampled_data, n_clusters)
    return pair_count


def full_bootstrap_procedure(
    data: pd.DataFrame, n_iterations: int, min_clusters: int = 2, max_clusters: int = 20
) -> dict:
    """Run the full bootstrap procedure."""
    full_result = defaultdict(int)
    for n_clusters in range(min_clusters, max_clusters + 1):
        logger.info(
            f"\n\nRunning bootstrap for {n_clusters} clusters:\n=============================================="
        )
        full_result[n_clusters] = bootstrap_procedure_for_one_set_of_settings(
            data, n_iterations, n_clusters
        )

    return full_result


def build_probability_matrix(
    pairing_probabilities: dict, columns: pd.Index
) -> pd.DataFrame:
    """Build a DataFrame from the pairing probabilities."""
    num_features = len(columns)
    probability_matrix = np.zeros((num_features, num_features))

    # DEBUGGING STATEMENTS
    logger.debug(f"| `build_probability_matrix` | Number of features: {num_features}")
    logger.debug(
        f"| `build_probability_matrix` | Probability matrix shape: {probability_matrix.shape}"
    )

    # Fill in the probability matrix
    for (i, j), prob in pairing_probabilities.items():
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
    return pd.DataFrame(probability_matrix, columns=columns, index=columns)


def varclus_bootstrap(
    data: pd.DataFrame,
    n_iterations: int = 100,
    min_clusters: int = 2,
    max_clusters: int = 20,
) -> pd.DataFrame:
    """Run bootstrap resampling and clustering to estimate variable clustering probabilities."""
    pair_count = full_bootstrap_procedure(
        data, n_iterations, min_clusters=2, max_clusters=20
    )
    logger.debug(
        f"| `varclus_bootstrap` | Dimensions of the full result: {len(pair_count)}"
    )
    pairing_probabilities = {
        n_clusters: calculate_pairing_probabilities(counts, n_iterations)
        for n_clusters, counts in pair_count.items()
    }

    # Combining results across all cluster numbers
    combined_probabilities = defaultdict(float)
    for cluster_probabilities in pairing_probabilities.values():
        for pair, prob in cluster_probabilities.items():
            combined_probabilities[pair] += prob / (max_clusters - min_clusters + 1)

    return build_probability_matrix(combined_probabilities, data.columns)
