import pytest
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from varclus.varclus__OLD import PCAHandler


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
def pca_handler(sample_data):
    return PCAHandler(sample_data)


def test_get_most_important_component(pca_handler):
    vars = ["A", "B"]
    component, variance = pca_handler.get_most_important_component(vars)
    assert component.shape == (2,), "Principal component should have 2 elements"
    assert 0 <= variance <= 1, "Explained variance ratio should be between 0 and 1"


def test_get_pca_variance(pca_handler):
    vars = ["A", "B"]
    pca_variance = pca_handler.get_pca_variance(vars)
    assert 0 <= pca_variance <= 1, "PCA variance should be between 0 and 1"


def test_invalid_vars(pca_handler):
    with pytest.raises(KeyError):
        pca_handler.get_most_important_component(["X", "Y"])


def test_empty_vars(pca_handler):
    with pytest.raises(ValueError):
        pca_handler.get_most_important_component([])


def test_single_variable(pca_handler):
    vars = ["A"]
    component, variance = pca_handler.get_most_important_component(vars)
    assert component.shape == (
        1,
    ), "Principal component should have 1 element for a single variable"
    assert 0 <= variance <= 1, "Explained variance ratio should be between 0 and 1"


def test_large_dataset():
    rng = np.random.default_rng(42)
    large_data = pd.DataFrame(rng.normal(0, 1, (1000, 50)))
    pca_handler = PCAHandler(large_data)
    vars = large_data.columns[:10].tolist()
    component, variance = pca_handler.get_most_important_component(vars)
    assert component.shape == (10,), "Principal component should have 10 elements"
    assert 0 <= variance <= 1, "Explained variance ratio should be between 0 and 1"
