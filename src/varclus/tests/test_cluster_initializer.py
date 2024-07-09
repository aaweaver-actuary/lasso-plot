import pytest
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from varclus.varclus__OLD import ClusterInitializer


@pytest.fixture
def sample_distance_matrix():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 1, (100, 5))
    return squareform(pdist(data, metric="euclidean"))


@pytest.fixture
def cluster_initializer(sample_distance_matrix):
    return ClusterInitializer(sample_distance_matrix, max_clusters=3)


def test_perform_hierarchical_clustering(cluster_initializer):
    Z = cluster_initializer.perform_hierarchical_clustering()
    assert Z.shape[1] == 4, "Linkage matrix should have 4 columns"
    assert (
        Z.shape[0] == cluster_initializer.dist_matrix.shape[0] - 1
    ), "Linkage matrix should have n-1 rows for n samples"


def test_get_clusters_max_clusters(cluster_initializer):
    Z = cluster_initializer.perform_hierarchical_clustering()
    clusters = cluster_initializer.get_clusters(Z)
    assert len(np.unique(clusters)) <= 3, "There should be at most 3 clusters"
    assert (
        clusters.shape[0] == cluster_initializer.dist_matrix.shape[0]
    ), "Cluster labels array should have same length as number of samples"


def test_get_clusters_threshold(sample_distance_matrix):
    cluster_initializer = ClusterInitializer(sample_distance_matrix, threshold=0.5)
    Z = cluster_initializer.perform_hierarchical_clustering()
    clusters = cluster_initializer.get_clusters(Z)
    assert (
        clusters.shape[0] == cluster_initializer.dist_matrix.shape[0]
    ), "Cluster labels array should have same length as number of samples"


def test_invalid_linkage_matrix(cluster_initializer):
    with pytest.raises(ValueError):
        cluster_initializer.get_clusters("invalid_linkage_matrix")
