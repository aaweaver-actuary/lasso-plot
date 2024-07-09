"""Bootstrap resampling for estimating variable clustering probabilities."""

import numpy as np
import pandas as pd
from collections import defaultdict
from varclus.varclus import VarClus
import logging

# Initialize the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
file_handler = logging.FileHandler("bootstrap_varclus.log")
file_handler.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

# Create formatters and add them to handlers
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def calculate_pairing_probabilities(pair_count: dict, num_runs: int) -> dict:
    """Calculate pairing probabilities from the pair count."""
    return {pair: count / num_runs for pair, count in pair_count.items()}


def resampled_data_draw(data: pd.DataFrame) -> pd.DataFrame:
    """Draw a resampled dataset from the input data."""
    return data.sample(n=len(data), replace=True, axis=0)


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


def full_bootstrap_procedure(
    data: pd.DataFrame, n_iterations: int, n_clusters: int
) -> dict:
    """Run the full bootstrap procedure."""
    pair_count = defaultdict(int)
    for _ in range(n_iterations):
        logger.info(f"Bootstrap iteration {_ + 1}/{n_iterations}")
        resampled_data = resampled_data_draw(data)
        pair_count = single_bootstrap_iteration(pair_count, resampled_data, n_clusters)
    return pair_count


def build_probability_matrix(
    pairing_probabilities: dict, columns: pd.Index
) -> pd.DataFrame:
    """Build a DataFrame from the pairing probabilities."""
    num_features = len(columns)
    probability_matrix = np.zeros((num_features, num_features))
    for (i, j), prob in pairing_probabilities.items():
        probability_matrix[i, j] = prob
        probability_matrix[j, i] = prob  # since the graph is undirected
    return pd.DataFrame(probability_matrix, columns=columns, index=columns)


def varclus_bootstrap(
    data: pd.DataFrame, n_iterations: int = 100, n_clusters: int = 10
) -> pd.DataFrame:
    """Run bootstrap resampling and clustering to estimate variable clustering probabilities."""
    pair_count = full_bootstrap_procedure(data, n_iterations, n_clusters)
    pairing_probabilities = calculate_pairing_probabilities(pair_count, n_iterations)
    return build_probability_matrix(pairing_probabilities, data.columns)
