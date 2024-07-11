"""Randomly-generate some classification data with 50 features, only 5 of which are informative. Use the generated data to perform integration testing on the VarClus class and the varclus_bootstrap function."""

from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import plotly.graph_objects as go
import networkx as nx
from varclus.varclus import VarclusBootstrapRunner, VarclusClusterer
from varclus.logger import logger
from varclus.src.anclust_runner import AnClustRunner, AnClustResampler
import itertools
from typing import Callable
from tqdm import tqdm


RANDOM_SEED = 42
METRICS = ["euclidean"]
LINKAGES = ["ward"]
POOLING_FUNCS = [np.mean]
N_CLUSTERS = list(range(2, 7))


# Generate the data
def data():
    X, y = make_classification(
        n_samples=10000,
        n_features=100,
        n_informative=10,
        n_redundant=80,
        n_clusters_per_class=2,
        random_state=RANDOM_SEED,
    )
    return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(100)])


# Integration test for the varclus_bootstrap function (plus interactive
# visualization to ensure the results are reasonable)
def test_varclus_bootstrap_algorithm(bootstrap_data):
    pair_count = {}

    for metric, linkage, pooling_func, n_clusters in tqdm(
        itertools.product(METRICS, LINKAGES, POOLING_FUNCS, N_CLUSTERS),
        desc="Running VarClus",
        total=len(METRICS) * len(LINKAGES) * len(POOLING_FUNCS) * len(N_CLUSTERS),
    ):
        logger.debug(
            f"Running VarClus with metric={metric}, linkage={linkage}, pooling_func={pooling_func}, n_clusters={n_clusters}"
        )
        vc = VarclusBootstrapRunner(
            data=bootstrap_data,
            runner=AnClustRunner(
                data=bootstrap_data,
                n_clusters=n_clusters,
                metric=metric,
                linkage=linkage,
                pooling_func=pooling_func,
            ),
            resampler=AnClustResampler(data=bootstrap_data, n_jobs=12),
            bootstrap_sample_fraction=0.75,
            n_observations=2000,
            n_iterations=100,
        )

        pair_count = vc.bootstrap_loop(pair_count)

    # for k, v in pair_count.items():
    #     pair_count[k] = v / vc.total_iterations

    count_matrix = pd.DataFrame(
        np.zeros((vc.num_features, vc.num_features)), columns=bootstrap_data.columns
    )

    for (i, j), p in pair_count.items():
        count_matrix.iloc[i, j] = p
        count_matrix.iloc[j, i] = p

    logger.debug(f"max count: {np.max(count_matrix)}")

    probability_matrix = count_matrix / np.max(count_matrix)

    return count_matrix, probability_matrix


# if __name__ == "__main__":
#     # Generate the data
#     bootstrap_data = data()

#     # Integration test for the varclus_bootstrap function
#     test_varclus_bootstrap_algorithm(bootstrap_data)

#     # fig = go.Figure(data=go.Heatmap(z=probability_matrix.values))
#     # fig.update_layout(title="Pairing probabilities between features")

#     # fig.write_html("varclus_bootstrap_heatmap.html")

#     # # Plot the force-directed graph
#     # vcc = VarclusClusterer(vc)
#     # clusters = vcc.cluster_features()
#     # fig = vcc.plot(probability_matrix, clusters)
#     # fig.write_html("varclus_bootstrap_force_directed_graph.html")
