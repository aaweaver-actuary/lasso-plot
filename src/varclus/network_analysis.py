"""Implements network/graph analysis functionality to analyze the relationships between variables in a dataset."""

import numpy as np
import pandas as pd
import networkx as nx
from scipy.cluster.hierarchy import linkage, fcluster
from collections import defaultdict
import matplotlib.pyplot as plt


def generate_synthetic_data(n_samples: int = 100, n_features: int = 10) -> pd.DataFrame:
    """Generate synthetic data for clustering analysis."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        rng.normal(size=(n_samples, n_features)),
        columns=[f"Feature_{i}" for i in range(n_features)],
    )


def run_clustering(data: pd.DataFrame, method: str="ward", num_clusters: int | None = None, distance_threshold: float | None = None) -> np.ndarray:
    """Run hierarchical clustering on the input data."""
    Z = linkage(data.T, method=method)
    if num_clusters:
        clusters = fcluster(Z, t=num_clusters, criterion="maxclust")
    elif distance_threshold:
        clusters = fcluster(Z, t=distance_threshold, criterion="distance")
    else:
        raise ValueError("Either num_clusters or distance_threshold must be provided.")
    return clusters


def count_variable_pairings(clusters, num_features):
    pair_count = defaultdict(int)
    for i in range(num_features):
        for j in range(i + 1, num_features):
            if clusters[i] == clusters[j]:
                pair_count[(i, j)] += 1
    return pair_count


def calculate_pairing_probabilities(pair_count, num_runs):
    pairing_probabilities = {
        pair: count / num_runs for pair, count in pair_count.items()
    }
    return pairing_probabilities


def bootstrap_resample(data):
    return data.sample(n=len(data), replace=True)


def create_adjacency_matrix(pairing_probabilities, num_features):
    adjacency_matrix = np.zeros((num_features, num_features))
    for (i, j), prob in pairing_probabilities.items():
        adjacency_matrix[i, j] = prob
        adjacency_matrix[j, i] = prob  # since the graph is undirected
    return adjacency_matrix


def analyze_graph(adjacency_matrix):
    G = nx.from_numpy_matrix(adjacency_matrix)

    degrees = dict(G.degree())
    weighted_degrees = dict(G.degree(weight="weight"))
    clustering_coefficients = nx.clustering(G, weight="weight")
    communities = list(
        nx.algorithms.community.greedy_modularity_communities(G, weight="weight")
    )

    return degrees, weighted_degrees, clustering_coefficients, communities


def plot_graph(G, probabilities):
    pos = nx.spring_layout(G)
    edges = G.edges(data=True)
    weights = [G[u][v]["weight"] for u, v in G.edges()]

    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color="lightblue")
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights)
    nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")
    plt.title("Feature Relationship Network")
    plt.show()


# # Parameters
# n_runs = 100
# num_clusters_list = [2, 3, 4, 5]
# distance_threshold_list = [0.5, 1.0, 1.5]

# # Generate synthetic data
# data = generate_synthetic_data()

# # Run clustering with bootstrap resampling and count pairings
# pair_count = defaultdict(int)
# num_features = data.shape[1]

# for _ in range(n_runs):
#     resampled_data = bootstrap_resample(data)
#     num_clusters = np.random.choice(num_clusters_list)
#     distance_threshold = np.random.choice(distance_threshold_list)
#     clusters = run_clustering(
#         resampled_data, num_clusters=num_clusters, distance_threshold=distance_threshold
#     )
#     pair_count.update(count_variable_pairings(clusters, num_features))

# # Calculate pairing probabilities
# pairing_probabilities = calculate_pairing_probabilities(pair_count, n_runs)

# # Create adjacency matrix
# adjacency_matrix = create_adjacency_matrix(pairing_probabilities, num_features)

# # Analyze the graph
# degrees, weighted_degrees, clustering_coefficients, communities = analyze_graph(
#     adjacency_matrix
# )

# # Create the graph from adjacency matrix
# G = nx.from_numpy_matrix(adjacency_matrix)

# # Plot the graph
# plot_graph(G, pairing_probabilities)
