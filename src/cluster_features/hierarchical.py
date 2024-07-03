"""Perform hierarchical clustering on a set of features."""

from __future__ import annotations
import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
from typing import Callable

# Specify the type of the distance function.
DistanceFunction = Callable[[np.ndarray, np.ndarray], float]


def compute_distance_matrix(
    data: np.ndarray, distance_func: DistanceFunction
) -> np.ndarray:
    """
    Compute the pairwise distance matrix using the given distance function.

    Parameters
    ----------
    data : np.ndarray
        2D array where rows are samples and columns are features.
    distance_func : DistanceFunction
        Function to compute the distance between two 1D arrays.

    Returns
    -------
    np.ndarray
        Pairwise distance matrix.
    """
    n_features = data.shape[1]
    distance_matrix = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(i + 1, n_features):
            distance = distance_func(data[:, i], data[:, j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    return distance_matrix


def hierarchical_clustering(
    data: np.ndarray,
    distance_func: DistanceFunction,
    method: str = "ward",
    criterion: str = "distance",
) -> np.ndarray:
    """Perform hierarchical clustering on a set of features.

    Parameters
    ----------
    data : np.ndarray
        2D array where rows are samples and columns are features.
    distance_func : DistanceFunction
        Function to compute the distance between two 1D arrays.
    method : str, optional
        The linkage method to use, by default "ward". Other options include
        "complete", "average", "weighted", "centroid", "median", and "ward".
    criterion : str, optional
        The criterion to use for clustering, by default "distance". Other
        options include "inconsistent", "maxclust", "monocrit", and "distance".

    Returns
    -------
    np.ndarray
        A 1D array of cluster labels for each feature.
    """
    distance_matrix = compute_distance_matrix(data, distance_func)
    linkage_matrix = linkage(squareform(distance_matrix), method=method)
    return fcluster(linkage_matrix, t=1.0, criterion=criterion)


def plot_dendrogram(
    data: np.ndarray,
    distance_func: DistanceFunction,
    method: str = "ward",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot the dendrogram of hierarchical clustering.

    Parameters
    ----------
    data : np.ndarray
        2D array where rows are samples and columns are features.
    distance_func : DistanceFunction
        Function to compute the distance between two 1D arrays.
    method : str, optional
        The linkage method to use, by default "ward". Other options include
        "complete", "average", "weighted", "centroid", "median", and "ward".
    ax : plt.Axes, optional
        The matplotlib axes to plot the dendrogram on. If None, a new figure
        and axes are created.

    Returns
    -------
    plt.Axes
        The matplotlib axes containing the dendrogram plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    distance_matrix = compute_distance_matrix(data, distance_func)
    linkage_matrix = linkage(squareform(distance_matrix), method=method)
    dendrogram(linkage_matrix, ax=ax)

    return ax
