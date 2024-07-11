"""A feature clustering algorithm based on the VarClus algorithm."""

from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from varclus.logger import logger
from varclus.interfaces import FeatureClusteringBootstrapRunner
import networkx as nx
import plotly.graph_objects as go


@dataclass
class VarclusClusterer:
    """A feature clustering algorithm based on the VarClus algorithm."""

    bootstrap_runner: FeatureClusteringBootstrapRunner
    method: str = "average"
    threshold: float = 0.7
    criterion: str = "distance"

    def run_bootstrap(self) -> None:
        """Run the bootstrap to calculate pairing probabilities."""
        logger.debug("Running the bootstrap to calculate pairing probabilities")
        self.bootstrap_runner.run_bootstrap()

    def read_probability_matrix(self) -> pd.DataFrame:
        """Read the pairing probabilities from disk."""
        try:
            return pd.read_parquet("pairing_probabilities.parquet")
        except FileNotFoundError:
            logger.error("Pairing probabilities not found. Running the bootstrap.")
            self.run_bootstrap()
            return pd.read_parquet("pairing_probabilities.parquet")

    def cluster_features(self) -> dict:
        """Cluster features based on their pairing probabilities."""
        probability_matrix = self.read_probability_matrix()
        logger.debug("Clustering features based on pairing probabilities")

        # Convert probability matrix to distance matrix
        logger.debug(
            "| `cluster_features` | Converting probability matrix to distance matrix"
        )
        distance_matrix = 1 - probability_matrix

        # Perform hierarchical clustering
        logger.debug("| `cluster_features` | Performing hierarchical clustering")
        linkage_matrix = linkage(squareform(distance_matrix), method=self.method)

        # Form clusters based on the given threshold
        logger.debug("| `cluster_features` | Forming clusters based on threshold")
        cluster_labels = fcluster(
            linkage_matrix, t=self.threshold, criterion=self.criterion
        )

        # Create a dictionary mapping features to their cluster labels
        logger.debug(
            "| `cluster_features` | Returning dictionary mapping features to cluster labels"
        )
        return {i: cluster_labels[i] for i in range(len(cluster_labels))}

    def plot(self, probability_matrix: pd.DataFrame, clusters: list[int]) -> go.Figure:
        """Plot a force-directed graph of the feature clusters."""
        # Create a graph
        G = nx.Graph()
        num_features = probability_matrix.shape[0]

        # Add nodes with cluster information
        for i in range(num_features):
            G.add_node(i, cluster=clusters[i])

        # Add edges with lengths proportional to 1 - probability
        for i in range(num_features):
            for j in range(i + 1, num_features):
                prob = probability_matrix.iloc[i, j]
                if prob > 0:  # Avoid adding edges with zero probability
                    G.add_edge(i, j, length=1 - prob)

        # Compute the positions using the spring layout
        pos = nx.spring_layout(G, weight="length")

        # Create edge traces
        edge_trace = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    line={"width": 0.5, "color": "#888"},
                    hoverinfo="none",
                    mode="lines",
                )
            )

        # Create node traces
        node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode="markers",
            hoverinfo="text",
            marker={
                "showscale": True,
                "colorscale": "YlGnBu",
                "color": [],
                "size": 10,
                "colorbar": {
                    "thickness": 15,
                    "title": "Cluster",
                    "xanchor": "left",
                    "titleside": "right",
                },
                "line_width": 2,
            },
        )

        for node in G.nodes(data=True):
            x, y = pos[node[0]]
            node_trace["x"] += (x,)
            node_trace["y"] += (y,)
            node_trace["marker"]["color"] += (node[1]["cluster"],)
            node_trace["text"] += (f"Feature {node[0]}",)

        fig = go.Figure(
            data=[*edge_trace, node_trace],
            layout=go.Layout(
                showlegend=False,
                hovermode="closest",
                margin={"b": 20, "l": 5, "r": 5, "t": 40},
                annotations=[
                    {
                        "text": "Force-directed graph of feature clusters",
                        "showarrow": False,
                        "xref": "paper",
                        "yref": "paper",
                    }
                ],
                xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
                yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            ),
        )

        return fig
