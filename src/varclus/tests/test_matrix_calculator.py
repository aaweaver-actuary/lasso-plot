import pytest
import pandas as pd
import numpy as np
from varclus.varclus__OLD import MatrixCalculator, handle_exceptions, validate_data


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
def matrix_calculator(sample_data):
    return MatrixCalculator(sample_data)


def test_compute_corr_matrix(matrix_calculator):
    corr_matrix = matrix_calculator.compute_corr_matrix()
    assert corr_matrix.shape == (4, 4), "Correlation matrix should be 4x4"
    assert np.allclose(
        np.diag(corr_matrix), 1
    ), "Diagonal elements of correlation matrix should be 1"
    assert np.all(
        (corr_matrix >= -1) & (corr_matrix <= 1)
    ), "Correlation matrix elements should be between -1 and 1"


def test_compute_dist_matrix(matrix_calculator):
    corr_matrix = matrix_calculator.compute_corr_matrix()
    dist_matrix = matrix_calculator.compute_dist_matrix(corr_matrix)
    assert dist_matrix.shape == (4, 4), "Distance matrix should be 4x4"
    assert np.all(
        (dist_matrix >= 0) & (dist_matrix <= 1)
    ), "Distance matrix elements should be between 0 and 1"


def test_empty_dataframe():
    empty_data = pd.DataFrame()
    with pytest.raises(ValueError):
        matrix_calculator = MatrixCalculator(empty_data)
        matrix_calculator.compute_corr_matrix()


def test_dataframe_with_nans():
    nan_data = pd.DataFrame({"A": [1, 2, np.nan], "B": [4, 5, 6]})
    with pytest.raises(ValueError):
        matrix_calculator = MatrixCalculator(nan_data)
        matrix_calculator.compute_corr_matrix()


def test_dataframe_with_single_variable():
    single_var_data = pd.DataFrame({"A": [1, 2, 3]})
    with pytest.raises(ValueError):
        matrix_calculator = MatrixCalculator(single_var_data)
        matrix_calculator.compute_corr_matrix()


def test_invalid_corr_matrix(matrix_calculator):
    with pytest.raises(TypeError):
        matrix_calculator.compute_dist_matrix("invalid_corr_matrix")
