"""Plotly force diagram to visualize the quality of the clusters."""

import plotly.graph_objects as go
import networkx as nx
import pandas as pd


def plot_force_diagram(probability_matrix: pd.DataFrame) -> go.Figure:
    # networkx plot of the distance matrix (eg 1 - probability_matrix) -- nodes are the features, edges are the distances, and colors are the clusters
    G = nx.Graph()
    for i in range(probability_matrix.shape[0]):
        G.add_node(i, cluster=i)
    for (i, j), prob in probability_matrix.items():
        G.add_edge(i, j, weight=1 - prob)

    pos = nx.spring_layout(G)
    edge_trace = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                line={"width": 1, "color": "#888"},
                hoverinfo="none",
                mode="lines",
            )
        )

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

    for node in G.nodes():
        x, y = pos[node]
        node_trace["x"] += (x,)
        node_trace["y"] += (y,)
        node_trace["marker"]["color"] += (G.nodes[node]["cluster"],)

    fig = go.Figure(data=[*edge_trace, node_trace], layout=go.Layout(showlegend=False))

    return fig
