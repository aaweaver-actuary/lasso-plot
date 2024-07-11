"""Randomly-generate some classification data with 50 features, only 5 of which are informative. Use the generated data to perform integration testing on the VarClus class and the varclus_bootstrap function."""

from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import plotly.graph_objects as go
import networkx as nx
from varclus.varclus import VarclusBootstrapRunner, VarclusClusterer, VarclusRunner


RANDOM_SEED = 42


# Generate the data
@pytest.fixture
def data():
    X, y = make_classification(
        n_samples=10000,
        n_features=100,
        n_informative=10,
        n_redundant=80,
        n_clusters_per_class=2,
        random_state=RANDOM_SEED,
    )
    return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(100)]), pd.Series(
        y, name="target"
    )


@pytest.fixture
def bootstrap_data(data: tuple[pd.DataFrame, pd.Series]):
    X, _ = data
    return X


# Integration test for the varclus_bootstrap function (plus interactive
# visualization to ensure the results are reasonable)
def test_varclus_bootstrap_algorithm(bootstrap_data):
    vc = VarclusBootstrapRunner(
        VarclusRunner(data=bootstrap_data, n_jobs=12),
        data=bootstrap_data,
        initial_sample_fraction=0.75,
        max_clusters=3,
        n_iterations=40,
    )
    probability_matrix = vc.run_bootstrap()

    fig = go.Figure(data=go.Heatmap(z=probability_matrix.values))
    fig.update_layout(title="Pairing probabilities between features")

    fig.write_html("varclus_bootstrap_heatmap.html")

    # Plot the force-directed graph
    vcc = VarclusClusterer(vc)
    clusters = vcc.cluster_features()
    fig = vcc.plot(probability_matrix, clusters)
    fig.write_html("varclus_bootstrap_force_directed_graph.html")
