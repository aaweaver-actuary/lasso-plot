import pytest
import pandas as pd
import numpy as np
from varclus.varclus_bootstrap import (
    calculate_pairing_probabilities,
    resampled_data_draw,
    update_pair_counts,
    single_bootstrap_iteration,
    full_bootstrap_procedure,
    build_probability_matrix,
    varclus_bootstrap,
)
from varclus.varclus import VarClus, run_pca, memory
from collections import defaultdict


# Generate synthetic data for testing
@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        rng.normal(size=(100, 10)), columns=[f"Feature_{i}" for i in range(10)]
    )


def test_run_pca(synthetic_data):
    """Test the PCA function."""
    chunk = synthetic_data.iloc[:, :5]
    result = run_pca(chunk)
    assert (
        result.shape == (1, 5)
    ), f"Expected shape (1, 5), but got {result.shape}. PCA might not have run correctly on chunk: {chunk.head()}."
    assert isinstance(
        result, np.ndarray
    ), f"Expected a numpy array, but got {type(result)}. Check the PCA implementation and the chunk data: {chunk.head()}."


def test_run_pca_single_feature(synthetic_data):
    """Test the PCA function on a single feature."""
    single_feature_chunk = synthetic_data.iloc[:, :1]
    result = run_pca(single_feature_chunk)
    assert result.shape == (
        1,
        1,
    ), f"Expected shape (1, 1) for single feature, but got {result.shape}."


def test_varclus_init(synthetic_data):
    """Test the initialization of the VarClus class."""
    varclus = VarClus(data=synthetic_data, n_clusters=3, n_jobs=2)
    assert varclus.data.equals(
        synthetic_data
    ), f"Expected {synthetic_data}, but got {varclus.data} instead."
    assert (
        varclus.n_clusters == 3
    ), f"Expected 3 clusters, but got {varclus.n_clusters}."
    assert varclus.n_jobs == 2, f"Expected 2 jobs, but got {varclus.n_jobs}."


def test_validate_data_correct(synthetic_data):
    """Test the validation of correct input data."""
    varclus = VarClus(data=synthetic_data)
    varclus.validate_data()


def test_validate_data_incorrect_type():
    """Test the validation of incorrect data type."""
    with pytest.raises(ValueError) as exc_info:
        varclus = VarClus(data=[1, 2, 3])
        varclus.validate_data()
    assert "Input data must be a pandas DataFrame." in str(
        exc_info.value
    ), f"Expected 'Input data must be a pandas DataFrame.' but got {exc_info.value!s}."


def test_validate_data_missing_values():
    """Test the validation of data with missing values."""
    data_with_nan = pd.DataFrame([[1, 2], [np.nan, 3]])
    with pytest.raises(ValueError) as exc_info:
        varclus = VarClus(data=data_with_nan)
        varclus.validate_data()
    assert "Input data contains missing values." in str(
        exc_info.value
    ), f"Expected 'Input data contains missing values.' but got {exc_info.value!s}."


def test_validate_data_empty():
    """Test the validation of empty data."""
    empty_data = pd.DataFrame()
    with pytest.raises(ValueError) as exc_info:
        varclus = VarClus(data=empty_data)
        varclus.validate_data()
    assert (
        "Input data must contain at least two variables." in str(exc_info.value)
    ), f"Expected 'Input data must contain at least two variables.' but got {exc_info.value!s}."


def test_run_clustering(synthetic_data):
    """Test the run_clustering method."""
    varclus = VarClus(data=synthetic_data, n_clusters=3)
    pca_result = run_pca(synthetic_data)
    clusters = varclus.run_clustering(pca_result)
    assert (
        len(clusters) == synthetic_data.shape[1]
    ), f"Expected {synthetic_data.shape[1]} clusters, but got {len(clusters)}."


def test_initialize_clusters(synthetic_data):
    """Test the initialization of clusters."""
    varclus = VarClus(data=synthetic_data, n_clusters=3, n_jobs=2)
    clusters = varclus.initialize_clusters()
    assert (
        len(clusters) == synthetic_data.shape[1]
    ), f"Expected {synthetic_data.shape[1]} clusters, but got {len(clusters)}."
    assert isinstance(
        clusters, np.ndarray
    ), f"Expected a numpy array, but got {type(clusters)}."


def test_run_varclus(synthetic_data):
    """Test the full run of the VarClus algorithm."""
    varclus = VarClus(data=synthetic_data, n_clusters=3, n_jobs=2)
    clusters = varclus.run()
    assert (
        len(clusters) == synthetic_data.shape[1]
    ), f"Expected {synthetic_data.shape[1]} clusters, but got {len(clusters)}."
    assert isinstance(
        clusters, np.ndarray
    ), f"Expected a numpy array, but got {type(clusters)}."


def test_bootstrap_resample(synthetic_data):
    """Test the bootstrap resampling and clustering."""
    probabilities_df = varclus_bootstrap(
        data=synthetic_data, n_iterations=10, n_clusters=3
    )
    assert isinstance(
        probabilities_df, pd.DataFrame
    ), f"Expected a DataFrame, but got {type(probabilities_df)}."
    assert (
        probabilities_df.shape == (synthetic_data.shape[1], synthetic_data.shape[1])
    ), f"Expected DataFrame shape {(synthetic_data.shape[1], synthetic_data.shape[1])}, but got {probabilities_df.shape}."
    assert all(
        probabilities_df.columns == synthetic_data.columns
    ), f"Expected DataFrame columns to be {list(synthetic_data.columns)}, but got {list(probabilities_df.columns)}."
    assert all(
        probabilities_df.index == synthetic_data.columns
    ), f"Expected DataFrame index to be {list(synthetic_data.columns)}, but got {list(probabilities_df.index)}."

    assert (
        probabilities_df.to_numpy() >= 0
    ).all(), "Expected all probabilities to be >= 0."
    assert (
        probabilities_df.to_numpy() <= 1
    ).all(), "Expected all probabilities to be <= 1."
    assert np.allclose(
        np.diag(probabilities_df.to_numpy()), 1
    ), "Expected diagonal values to be 1."


def test_memory_caching(synthetic_data):
    """Test the memory caching of the PCA function."""
    chunk = synthetic_data.iloc[:, :5]
    result1 = run_pca(chunk)
    result2 = run_pca(chunk)
    assert np.array_equal(
        result1, result2
    ), f"Expected cached results to be identical, but got different results: {result1} and {result2}."


def test_perfectly_correlated_features():
    """Test the clustering with perfectly correlated features."""
    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        {
            "Feature_1": rng.normal(size=100),
            "Feature_2": rng.normal(size=100),
            "Feature_3": rng.normal(size=100),
            "Feature_4": rng.normal(size=100),
            "Feature_5": rng.normal(size=100),
        }
    )
    data["Feature_6"] = data["Feature_1"] * 2
    data["Feature_7"] = data["Feature_2"] * -1
    data["Feature_8"] = data["Feature_3"] + 1
    data["Feature_9"] = data["Feature_4"] - 0.5
    data["Feature_10"] = data["Feature_5"] * 0.5

    probabilities_df = varclus_bootstrap(data=data, n_iterations=10, n_clusters=3)
    assert isinstance(
        probabilities_df, pd.DataFrame
    ), f"Expected a DataFrame, but got {type(probabilities_df)}."
    assert (
        probabilities_df.shape == (data.shape[1], data.shape[1])
    ), f"Expected DataFrame shape {(data.shape[1], data.shape[1])}, but got {probabilities_df.shape}."


def test_varclus_parameter_bounds(synthetic_data):
    """Test the VarClus with parameter bounds."""
    with pytest.raises(ValueError) as exc_info:
        varclus = VarClus(data=synthetic_data, n_clusters=synthetic_data.shape[1] + 1)
        varclus.run()
    assert (
        "Number of clusters cannot exceed the number of features" in str(exc_info.value)
        or "number sections must be larger than 0." in str(exc_info.value)
    ), f"Expected 'Number of clusters cannot exceed the number of features' or 'number sections must be larger than 0.' but got {exc_info.value!s}."


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
    expected_matrix = np.array(
        [
            [0.0, 0.5, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.3, 0.0],
            [0.0, 0.0, 0.3, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    result = build_probability_matrix(pairing_probabilities, num_features)
    assert np.allclose(
        result.values, expected_matrix
    ), f"Expected matrix:\n{expected_matrix}\nBut got:\n{result.to_numpy()}"


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


# Additional tests for edge cases
def test_varclus_bootstrap_single_feature():
    """Test the varclus_bootstrap function with a single feature."""
    rng = np.random.default_rng(42)
    data = pd.DataFrame({"Feature_1": rng.normal(size=100)})
    result = varclus_bootstrap(data, n_iterations=10, n_clusters=1)
    assert isinstance(
        result, pd.DataFrame
    ), f"Expected a DataFrame, but got {type(result)}."
    assert result.shape == (1, 1), f"Expected shape (1, 1), but got {result.shape}."
    assert (
        result.iloc[0, 0] == 1.0
    ), f"Expected the diagonal value to be 1, but got {result.iloc[0, 0]}."


def test_varclus_bootstrap_more_clusters_than_features(synthetic_data):
    """Test the varclus_bootstrap function with more clusters than features."""
    with pytest.raises(ValueError):
        varclus_bootstrap(
            synthetic_data, n_iterations=10, n_clusters=synthetic_data.shape[1] + 1
        )


def test_varclus_invalid_data_type():
    """Test VarClus with invalid data type."""
    with pytest.raises(ValueError) as exc_info:
        varclus = VarClus(data=[1, 2, 3])
        varclus.validate_data()
    assert "Input data must be a pandas DataFrame." in str(exc_info.value)


def test_varclus_missing_values():
    """Test VarClus with missing values in data."""
    data_with_nan = pd.DataFrame([[1, 2], [np.nan, 3]])
    with pytest.raises(ValueError) as exc_info:
        varclus = VarClus(data=data_with_nan)
        varclus.validate_data()
    assert "Input data contains missing values." in str(exc_info.value)


def test_varclus_empty_data():
    """Test VarClus with empty data."""
    empty_data = pd.DataFrame()
    with pytest.raises(ValueError) as exc_info:
        varclus = VarClus(data=empty_data)
        varclus.validate_data()
    assert "Input data must contain at least two variables." in str(exc_info.value)


def test_varclus_single_column():
    """Test VarClus with single column data."""
    rng = np.random.default_rng(42)
    single_column_data = pd.DataFrame({"Feature_1": rng.normal(size=100)})
    with pytest.raises(ValueError) as exc_info:
        varclus = VarClus(data=single_column_data)
        varclus.validate_data()
    assert "Input data must contain at least two variables." in str(exc_info.value)


def test_varclus_too_many_clusters(synthetic_data):
    """Test VarClus with more clusters than features."""
    with pytest.raises(ValueError) as exc_info:
        varclus = VarClus(data=synthetic_data, n_clusters=synthetic_data.shape[1] + 1)
        varclus.run()
    assert "Number of clusters cannot exceed the number of features" in str(
        exc_info.value
    ) or "Cannot extract more clusters than samples" in str(exc_info.value)


def test_resampled_data_draw_minimal():
    """Test resampling with minimal data."""
    rng = np.random.default_rng(42)
    minimal_data = pd.DataFrame(
        {"Feature_1": rng.standard_normal(5), "Feature_2": rng.standard_normal(5)}
    )
    resampled_data = resampled_data_draw(minimal_data)
    assert (
        resampled_data.shape == minimal_data.shape
    ), f"Expected shape {minimal_data.shape}, but got {resampled_data.shape}."


def test_single_bootstrap_iteration_minimal():
    """Test single bootstrap iteration with minimal data."""
    rng = np.random.default_rng(42)
    minimal_data = pd.DataFrame(
        {"Feature_1": rng.standard_normal(5), "Feature_2": rng.standard_normal(5)}
    )
    pair_count = defaultdict(int)
    result = single_bootstrap_iteration(pair_count, minimal_data, n_clusters=2)
    assert isinstance(
        result, defaultdict
    ), f"Expected a defaultdict, but got {type(result)}."
    assert len(result) > 0, "Expected non-empty pair counts."


def test_full_bootstrap_procedure_minimal():
    """Test full bootstrap procedure with minimal data."""
    rng = np.random.default_rng(42)
    minimal_data = pd.DataFrame(
        {"Feature_1": rng.standard_normal(5), "Feature_2": rng.standard_normal(5)}
    )
    n_iterations = 5
    n_clusters = 2
    result = full_bootstrap_procedure(minimal_data, n_iterations, n_clusters)
    assert isinstance(
        result, defaultdict
    ), f"Expected a defaultdict, but got {type(result)}."
    assert len(result) > 0, "Expected non-empty pair counts."


def test_varclus_bootstrap_minimal():
    """Test varclus_bootstrap with minimal data."""
    rng = np.random.default_rng(42)
    minimal_data = pd.DataFrame(
        {"Feature_1": rng.standard_normal(5), "Feature_2": rng.standard_normal(5)}
    )
    result = varclus_bootstrap(minimal_data, n_iterations=5, n_clusters=2)
    assert isinstance(
        result, pd.DataFrame
    ), f"Expected a DataFrame, but got {type(result)}."
    assert (
        result.shape == (minimal_data.shape[1], minimal_data.shape[1])
    ), f"Expected shape {(minimal_data.shape[1], minimal_data.shape[1])}, but got {result.shape}."
    assert (result.to_numpy() >= 0).all(), "Expected all probabilities to be >= 0."
    assert (result.to_numpy() <= 1).all(), "Expected all probabilities to be <= 1."
