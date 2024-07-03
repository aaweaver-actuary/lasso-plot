"""Implement various distance metrics for clustering algorithms."""

from __future__ import annotations
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import mahalanobis
import itertools

import polars as pl
import polars.selectors as cs

__all__ = [
    "euclidean_distance",
    "manhattan_distance",
    "cosine_similarity",
    "pearson_correlation",
    "jaccard_similarity",
    "hamming_distance",
    "mahalanobis_distance",
]


def __handle_polars_input(df: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    """Ensure that the input is a LazyFrame."""
    if isinstance(df, pl.DataFrame):
        return df.lazy()
    return df


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate the Euclidean distance between two points."""
    return np.sqrt(np.sum((x - y) ** 2))


def manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate the Manhattan distance between two points.

    Manhattan distance is the sum of the absolute differences of their
    coordinates. It is also known as the L1 distance, taxicab distance,
    rectilinear distance, or city block distance.

    Manhattan distance is less sensitive to outliers than the Euclidean
    distance, and it is often used in clustering algorithms because it
    is less affected by high-dimensional data.
    """
    return np.sum(np.abs(x - y))


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate the cosine similarity between two vectors x and y.

    Cosine similarity is a measure of similarity between two non-zero
    vectors of an inner product space. It is defined to equal the cosine
    of the angle between them, which is also the same as the inner product
    of the same vectors normalized to both have length 1.

    Cosine similarity is particularly used in positive space, where the
    outcome is neatly bounded in [0, 1].

    Cosine similarity is a judgment of orientation and not magnitude: two
    vectors with the same orientation have a cosine similarity of 1, two
    vectors at 90° have a similarity of 0, and two vectors diametrically
    opposed have a similarity of -1, independent of their magnitude.

    Parameters
    ----------
    x, y : array-like of shape (n_features,)
        Input vectors.

    Returns
    -------
    float
        The cosine similarity between vectors x and y.
    """
    return np.dot(x, y) / (norm(x) * norm(y))


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate the Pearson correlation coefficient between two arrays.

    The Pearson correlation coefficient measures the linear relationship
    between two datasets. Strictly speaking, Pearson's correlation requires
    that each dataset be normally distributed, and it is not robust against
    outliers.

    Pearson correlation is often used in feature selection techniques to
    identify the most relevant features in a dataset. By calculating the
    correlation between each feature and the target variable, we can
    determine which features are most likely to have a linear relationship
    with the target.

    In a clustering context, Pearson correlation can be used to determine
    the similarity between two data points. If two data points have a high
    correlation, they are likely to be close to each other in the feature
    space.

    To perform clustering, you would typically calculate the Pearson
    correlation between each pair of data points in the dataset and use
    these values to construct a similarity matrix. This similarity matrix
    can then be used as input to a clustering algorithm to group similar
    data points together.

    Parameters
    ----------
    x, y : array-like of shape (n_features,)
        Input arrays.

    Returns
    -------
    float
        The Pearson correlation coefficient between arrays x and y.
    """
    return np.corrcoef(x, y)[0, 1]


def jaccard_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate the Jaccard similarity between two arrays.

    The Jaccard similarity coefficient is a measure of similarity between
    two sets. It is defined as the size of the intersection divided by the
    size of the union of the two sets.

    Jaccard similarity is often used in clustering algorithms to determine
    the similarity between two data points. If two data points have a high
    Jaccard similarity, they are likely to be close to each other in the
    feature space.

    To perform clustering, you would typically calculate the Jaccard
    similarity between each pair of data points in the dataset and use
    these values to construct a similarity matrix. This similarity matrix
    can then be used as input to a clustering algorithm to group similar
    data points together.

    Parameters
    ----------
    x, y : array-like of shape (n_features,)
        Input arrays.

    Returns
    -------
    float
        The Jaccard similarity between arrays x and y.
    """
    intersection = np.logical_and(x, y)
    union = np.logical_or(x, y)

    # Avoid division by zero
    return np.sum(intersection) / np.sum(union) if np.sum(union) != 0 else 0.0


def hamming_distance(x: np.ndarray, y: np.ndarray) -> int:
    """Calculate the Hamming distance between two arrays.

    The Hamming distance is a measure of the difference between two strings
    of equal length. It is the number of positions at which the corresponding
    symbols are different.

    In a clustering context, the Hamming distance can be used to determine
    the similarity between two data points. If two data points have a low
    Hamming distance, they are likely to be close to each other in the
    feature space.

    Parameters
    ----------
    x, y : array-like of shape (n_features,)
        Input arrays.

    Returns
    -------
    int
        The Hamming distance between arrays x and y.
    """
    return np.sum(x != y)


def mahalanobis_distance(x: np.ndarray, y: np.ndarray, data: np.ndarray) -> float:
    """Calculate the Mahalanobis distance between two points.

    The Mahalanobis distance is a measure of the distance between a point
    and a distribution. It is a multi-dimensional generalization of the
    idea of measuring how many standard deviations away a point is from the
    mean of a distribution.

    The Mahalanobis distance is used in many statistical applications,
    including outlier detection, clustering, and classification. It is
    especially useful when the data is not normally distributed or when
    the features are correlated.

    Parameters
    ----------
    x, y : array-like of shape (n_features,)
        Input arrays.
    data : array-like of shape (n_samples, n_features)
        The data used to calculate the covariance matrix.

    Returns
    -------
    float
        The Mahalanobis distance between arrays x and y.
    """
    covariance_matrix = np.cov(data.T)
    inv_cov_matrix = np.linalg.inv(covariance_matrix)
    return mahalanobis(x, y, inv_cov_matrix)
