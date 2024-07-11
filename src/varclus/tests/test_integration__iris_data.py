import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris
from varclus.src._bootstrap import varclus_bootstrap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


@pytest.fixture
def cancer_data():
    """Load the breast cancer dataset and return as a DataFrame."""
    cancer = load_breast_cancer()
    df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names).assign(
        target=cancer.target
    )
    df.columns = df.columns.str.replace(" ", "_")
    return df


@pytest.fixture
def iris_data():
    """Load the Iris dataset and return as a DataFrame."""
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names).assign(
        target=iris.target
    )
    df.columns = df.columns.str.replace(" ", "_").str.replace("_(cm)", "")

    # add some interaction terms for testing
    df["sepal_length_width"] = df["sepal_length"] * df["sepal_width"]
    df["petal_length_width"] = df["petal_length"] * df["petal_width"]
    df["sepal_petal_length"] = df["sepal_length"] * df["petal_length"]
    df["sepal_petal_width"] = df["sepal_width"] * df["petal_width"]
    df["sepal_length^2"] = df["sepal_length"] ** 2
    df["sepal_width^2"] = df["sepal_width"] ** 2
    df["petal_length^2"] = df["petal_length"] ** 2
    df["petal_width^2"] = df["petal_width"] ** 2
    return df


def plot_scatter_matrix_with_probabilities(
    data: pd.DataFrame, probabilities: pd.DataFrame
):
    """Create a Plotly scatter matrix plot next to the matrix of probabilities.

    Parameters
    ----------
    data: pd.DataFrame, the input data.
    probabilities: pd.DataFrame, the matrix of clustering probabilities.

    Returns
    -------
    fig: go.Figure, the combined figure with the scatter matrix and probability matrix.

    """
    # Define indices corresponding to flower categories, using pandas label encoding
    index_vals = data["target"].astype("category")

    data = data.drop(columns="target")

    # Create the scatter matrix plot
    # scatter_matrix = go.Splom(
    #     dimensions=[{"label": col, "values": data[col]} for col in data.columns],
    #     marker={
    #         "color": index_vals,
    #         "showscale": False,  # colors encode categorical variables
    #         "line_color": "black",
    #         "line_width": 0.5,
    #     },
    # )

    scatter_fig = px.scatter_matrix(
        data,
        dimensions=data.columns,
        color=index_vals,
        title="Scatter Matrix",
        labels={col: col.replace("_", " ") for col in data.columns},
    )

    scatter_fig.write_html("scatter_matrix.html")

    # Create the heatmap for the probability matrix
    heatmap = go.Heatmap(
        z=probabilities.values,
        x=probabilities.columns,
        y=probabilities.index,
        colorscale="Viridis",
        colorbar={"title": "Probability"},
    )

    heatmap_fig = go.Figure(heatmap)
    heatmap_fig.update_layout(
        title="Variable Clustering Probabilities",
        xaxis_title="Feature",
        yaxis_title="Feature",
    )

    heatmap_fig.write_html("heatmap.html")


def test_varclus_bootstrap_integration(iris_data):
    """Integration test for the varclus_bootstrap function using Iris dataset."""
    n_iterations = 1000
    n_clusters = 2
    # result = varclus_bootstrap(
    #     cancer_data.drop(columns="target"), n_iterations, n_clusters
    # )

    result_iris = varclus_bootstrap(
        iris_data.drop(columns="target"), n_iterations, n_clusters
    )

    # # Basic checks on the result
    # assert isinstance(
    #     result, pd.DataFrame
    # ), f"Expected a DataFrame, but got {type(result)}."
    # assert (
    #     result.shape == (cancer_data.shape[1] - 1, cancer_data.shape[1] - 1)
    # ), f"Expected shape {(cancer_data.shape[1] - 1, cancer_data.shape[1] - 1)}, but got {result.shape}."
    # assert (result.to_numpy() >= 0).all(), "Expected all probabilities to be >= 0."
    # assert (result.to_numpy() <= 1).all(), "Expected all probabilities to be <= 1."

    # Plot the scatter matrix with probabilities
    plot_scatter_matrix_with_probabilities(iris_data, result_iris)
