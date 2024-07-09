import pytest
import pandas as pd
import numpy as np
from collections import defaultdict
from varclus.varclus_bootstrap import (
    calculate_pairing_probabilities,
    resampled_data_draw,
    update_pair_counts,
    single_bootstrap_iteration,
    full_bootstrap_procedure,
    build_probability_matrix,
    varclus_bootstrap,
)
from varclus.varclus import VarClus, run_pca


# Generate synthetic data for testing
@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        rng.normal(size=(100, 10)), columns=[f"Feature_{i}" for i in range(10)]
    )


def test_calculate_pairing_probabilities():
    """Test the calculation of pairing probabilities."""
    pair_count = {(0, 1): 5, (2, 3): 3, (4, 5): 2}
    num_runs = 10
    expected_probabilities = {(0, 1): 0.5, (2, 3): 0.3, (4, 5): 0.2}
    result = calculate_pairing_probabilities(pair_count, num_runs)
    assert (
        result == expected_probabilities
    ), f"Expected {expected_probabilities}, but got {result}."


def test_resampled_data_draw(synthetic_data):
    """Test resampling data."""
    resampled_data = resampled_data_draw(synthetic_data)
    assert (
        resampled_data.shape == synthetic_data.shape
    ), f"Expected shape {synthetic_data.shape}, but got {resampled_data.shape}."
    assert resampled_data.columns.equals(
        synthetic_data.columns
    ), f"Expected columns {synthetic_data.columns}, but got {resampled_data.columns}."


def test_update_pair_counts():
    """Test updating pair counts."""
    clusters = np.array([0, 1, 0, 1, 0])
    pair_count = defaultdict(int)
    expected_pair_count = {(0, 2): 1, (0, 4): 1, (2, 4): 1, (1, 3): 1}
    result = update_pair_counts(pair_count, clusters)
    assert (
        result == expected_pair_count
    ), f"Expected {expected_pair_count}, but got {result}."


def test_single_bootstrap_iteration(synthetic_data):
    """Test a single bootstrap iteration."""
    pair_count = defaultdict(int)
    n_clusters = 3
    result = single_bootstrap_iteration(pair_count, synthetic_data, n_clusters)
    assert isinstance(
        result, defaultdict
    ), f"Expected a defaultdict, but got {type(result)}."


def test_full_bootstrap_procedure(synthetic_data):
    """Test the full bootstrap procedure."""
    n_iterations = 10
    n_clusters = 3
    result = full_bootstrap_procedure(synthetic_data, n_iterations, n_clusters)
    assert isinstance(
        result, defaultdict
    ), f"Expected a defaultdict, but got {type(result)}."
    assert len(result) > 0, "Expected the result to have at least one entry."


def test_build_probability_matrix():
    """Test building the probability matrix."""
    pairing_probabilities = {(0, 1): 0.5, (2, 3): 0.3}
    num_features = 5
    columns = pd.Index([f"Feature_{i}" for i in range(num_features)])
    expected_matrix = pd.DataFrame(
        np.zeros((num_features, num_features)), columns=columns, index=columns
    )
    expected_matrix.iloc[0, 1] = 0.5
    expected_matrix.iloc[1, 0] = 0.5
    expected_matrix.iloc[2, 3] = 0.3
    expected_matrix.iloc[3, 2] = 0.3

    result = build_probability_matrix(pairing_probabilities, columns)
    pd.testing.assert_frame_equal(result, expected_matrix)


def test_varclus_bootstrap(synthetic_data):
    """Test the varclus_bootstrap function."""
    n_iterations = 10
    n_clusters = 3
    result = varclus_bootstrap(synthetic_data, n_iterations, n_clusters)
    assert isinstance(
        result, pd.DataFrame
    ), f"Expected a DataFrame, but got {type(result)}."
    assert (
        result.shape == (synthetic_data.shape[1], synthetic_data.shape[1])
    ), f"Expected shape {(synthetic_data.shape[1], synthetic_data.shape[1])}, but got {result.shape}."
    assert all(
        result.columns == synthetic_data.columns
    ), f"Expected columns {synthetic_data.columns}, but got {result.columns}."
    assert all(
        result.index == synthetic_data.columns
    ), f"Expected index {synthetic_data.columns}, but got {result.index}."
    assert (result.to_numpy() >= 0).all(), "Expected all probabilities to be >= 0."
    assert (result.to_numpy() <= 1).all(), "Expected all probabilities to be <= 1."


def test_run_pca_single_feature(synthetic_data):
    """Test the PCA function on a single feature."""
    single_feature_chunk = synthetic_data.iloc[:, :1]
    result = run_pca(single_feature_chunk)
    assert result.shape == (
        1,
        1,
    ), f"Expected shape (1, 1) for single feature, but got {result.shape}."


def test_varclus_parameter_bounds(synthetic_data):
    """Test the VarClus with parameter bounds."""
    with pytest.raises(ValueError) as exc_info:
        varclus = VarClus(data=synthetic_data, n_clusters=synthetic_data.shape[1] + 1)
        varclus.run()
    assert (
        "Number of clusters cannot exceed the number of features" in str(exc_info.value)
        or "Cannot extract more clusters than samples" in str(exc_info.value)
    ), f"Expected 'Number of clusters cannot exceed the number of features' or 'Cannot extract more clusters than samples' but got {exc_info.value!s}."


def test_varclus_bootstrap_single_feature():
    """Test the varclus_bootstrap function with a single feature."""
    rng = np.random.default_rng(42)
    data = pd.DataFrame({"Feature_1": rng.normal(size=100)})
    with pytest.raises(ValueError) as exc_info:
        result = varclus_bootstrap(data, n_iterations=10, n_clusters=1)
    assert (
        "Input data must contain at least two variables." in str(exc_info.value)
    ), f"Expected 'Input data must contain at least two variables.' but got {exc_info.value!s}."
