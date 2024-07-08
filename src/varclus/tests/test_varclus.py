import pytest
import pandas as pd
import numpy as np
from varclus.varclus import (
    VarClus,
    MatrixCalculator,
    ClusterInitializer,
    PCAHandler,
    ClusterMergerSplitter,
)

import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filemode="w",
    filename="varclus_testing.log",
)

logger = logging.getLogger(__name__)
logger.debug("Testing VarClus module -- testing logger")
logging.debug("Testing VarClus module -- testing logging")


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
def varclus_instance(sample_data):
    return VarClus(sample_data, max_clusters=2)


def test_matrix_calculator(sample_data):
    calculator = MatrixCalculator(sample_data)
    corr_matrix = calculator.compute_corr_matrix()
    assert corr_matrix.shape == (4, 4), "Correlation matrix should be 4x4"
    assert np.allclose(
        np.diag(corr_matrix), 1
    ), "Diagonal elements of correlation matrix should be 1"

    dist_matrix = calculator.compute_dist_matrix(corr_matrix)
    assert dist_matrix.shape == (4, 4), "Distance matrix should be 4x4"
    assert np.all(
        (dist_matrix >= 0) & (dist_matrix <= 1)
    ), "Distance matrix elements should be between 0 and 1"


def test_cluster_initializer(sample_data):
    calculator = MatrixCalculator(sample_data)
    corr_matrix = calculator.compute_corr_matrix()
    dist_matrix = calculator.compute_dist_matrix(corr_matrix)
    initializer = ClusterInitializer(dist_matrix, max_clusters=2)

    Z = initializer.perform_hierarchical_clustering()
    assert Z.shape[1] == 4, "Linkage matrix should have 4 columns"

    clusters = initializer.get_clusters(Z)
    assert len(np.unique(clusters)) <= 2, "There should be at most 2 clusters"


def test_pca_handler(sample_data):
    pca_handler = PCAHandler(sample_data)
    vars = ["A", "B"]
    component, variance = pca_handler.get_most_important_component(vars)
    assert (
        component.shape == (2,)
    ), f"Principal component should have 2 elements, found {component.shape[0]} elements."
    assert (
        0 <= variance <= 1
    ), f"Explained variance ratio should be between 0 and 1, found {variance}."

    pca_variance = pca_handler.get_pca_variance(vars)
    assert (
        0 <= pca_variance <= 1
    ), f"PCA variance should be between 0 and 1, found {pca_variance}."


def test_varclus_initialization(varclus_instance):
    varclus_instance.initialize_clusters()
    clusters = varclus_instance.get_clusters()
    assert not clusters.empty, "Clusters should not be empty"
    assert (
        "Cluster" in clusters.columns
    ), "Clusters DataFrame should have a 'Cluster' column"


def test_varclus_run(varclus_instance):
    clustered_data = varclus_instance.run()
    logger.debug(f"Clustered data:\n\n{clustered_data}")

    assert not clustered_data.empty, "The result should not be empty"
    assert (
        "Cluster" in clustered_data.columns
    ), "The result should contain 'Cluster' column"
    assert (
        clustered_data["Cluster"].nunique() <= 2
    ), f"There should be at most 2 clusters, found {clustered_data['Cluster'].nunique()} clusters"


def test_varclus_edge_cases():
    # Empty DataFrame
    empty_data = pd.DataFrame()
    with pytest.raises(ValueError):
        varclus = VarClus(empty_data)
        varclus.initialize_clusters()

    # DataFrame with NaNs
    nan_data = pd.DataFrame({"A": [1, 2, np.nan], "B": [4, 5, 6]})
    with pytest.raises(ValueError):
        varclus = VarClus(nan_data)
        varclus.initialize_clusters()

    # DataFrame with a single variable
    single_var_data = pd.DataFrame({"A": [1, 2, 3]})
    with pytest.raises(ValueError):
        varclus = VarClus(single_var_data)
        varclus.initialize_clusters()


def test_varclus_large_dataset():
    rng = np.random.default_rng(42)
    large_data = pd.DataFrame(rng.normal(0, 1, (1000, 50)))
    varclus = VarClus(large_data, max_clusters=5)
    clustered_data = varclus.run()
    assert not clustered_data.empty, "The result should not be empty"
    assert (
        "Cluster" in clustered_data.columns
    ), "The result should contain 'Cluster' column"
    assert (
        clustered_data["Cluster"].nunique() <= 5
    ), "There should be at most 5 clusters"


def test_varclus_custom_threshold(sample_data):
    varclus = VarClus(sample_data, threshold=0.5)
    clustered_data = varclus.run()
    assert not clustered_data.empty, "The result should not be empty"
    assert (
        "Cluster" in clustered_data.columns
    ), "The result should contain 'Cluster' column"


def test_cluster_merger_splitter(sample_data):
    varclus = VarClus(sample_data, max_clusters=2)
    varclus.initialize_clusters()
    pca_handler = PCAHandler(sample_data)
    cluster_merger_splitter = ClusterMergerSplitter(
        sample_data, varclus.clusters, pca_handler
    )

    clusters = varclus.clusters
    unique_clusters = np.unique(clusters)

    for cluster in unique_clusters:
        cluster_merger_splitter.split_cluster(cluster)
        assert len(np.unique(varclus.clusters)) >= len(
            unique_clusters
        ), "Number of clusters should not decrease after splitting"
