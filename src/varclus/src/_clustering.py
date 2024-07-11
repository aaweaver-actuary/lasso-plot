"""Take the pairing probabilities from the varclus bootstrap and group features into clusters, trying to maximize the average probability within clusters."""

import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from varclus.logger import logger

__all__ = ["cluster_features"]


def cluster_features(
    probability_matrix: pd.DataFrame, method: str = "average", threshold: float = 0.7
) -> dict:
    """Cluster features based on their pairing probabilities.

    Parameters
    ----------
    probability_matrix : pd.DataFrame
        DataFrame containing the pairing probabilities between features.
    method : str, optional
        The linkage method to use for clustering (default is 'average').
    threshold : float, optional
        The threshold for forming clusters (default is 0.7).

    Returns
    -------
    dict
        Dictionary mapping feature indices to their cluster labels.
    """
    # Convert probability matrix to distance matrix
    logger.debug(
        "| `cluster_features` | Converting probability matrix to distance matrix"
    )
    distance_matrix = 1 - probability_matrix

    # Perform hierarchical clustering
    logger.debug("| `cluster_features` | Performing hierarchical clustering")
    linkage_matrix = linkage(squareform(distance_matrix), method=method)

    # Form clusters based on the given threshold
    logger.debug("| `cluster_features` | Forming clusters based on threshold")
    cluster_labels = fcluster(linkage_matrix, t=threshold, criterion="distance")

    # Create a dictionary mapping features to their cluster labels
    logger.debug(
        "| `cluster_features` | Returning dictionary mapping features to cluster labels"
    )
    return {i: cluster_labels[i] for i in range(len(cluster_labels))}
