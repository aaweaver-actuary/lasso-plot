import pytest
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from varclus.varclus__OLD import ClusterMergerSplitter, PCAHandler


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "A": rng.normal(0, 1, 100),
            "B": rng.normal(0, 1, 100),
            "C": rng.normal(0, 1, 100),
            "D": rng.normal(0, 1, 100),
        }
    )


@pytest.fixture
def clusters():
    return np.array([1, 1, 2, 2])


@pytest.fixture
def pca_handler(sample_data):
    return PCAHandler(sample_data)


@pytest.fixture
def cluster_merger_splitter(sample_data, clusters, pca_handler):
    return ClusterMergerSplitter(sample_data, clusters, pca_handler)


def test_get_variables(cluster_merger_splitter):
    cluster_vars = cluster_merger_splitter.get_variables(1)
    assert set(cluster_vars) == {
        "A",
        "B",
    }, "Variables in cluster 1 should be 'A' and 'B'"
    cluster_vars = cluster_merger_splitter.get_variables(2)
    assert set(cluster_vars) == {
        "C",
        "D",
    }, "Variables in cluster 2 should be 'C' and 'D'"


def test_should_two_clusters_merge(cluster_merger_splitter):
    result = cluster_merger_splitter.should_two_clusters_merge(1, 2)
    assert isinstance(
        result, bool | np.bool_
    ), f"Result should be a boolean, got {result} ({type(result)})."


def test_split_cluster(cluster_merger_splitter):
    cluster_merger_splitter.split_cluster(1)
    assert (
        len(np.unique(cluster_merger_splitter.clusters)) >= 2
    ), "Number of clusters should not decrease after splitting"


def test_split_single_variable_cluster(cluster_merger_splitter):
    clusters = np.array([1, 1, 1, 1])
    cluster_merger_splitter.clusters = clusters
    cluster_merger_splitter.split_cluster(1)

    assert not cluster_merger_splitter.should_two_clusters_merge(
        1, 1
    ), "Cannot merge a cluster with a single variable"


def test_invalid_cluster_label(cluster_merger_splitter):
    with pytest.raises(ValueError):
        cluster_merger_splitter.get_variables(3)
