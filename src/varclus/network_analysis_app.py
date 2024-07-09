"""Create a Dash app to visualize a network graph using Cytoscape."""

import dash
from dash import html
import dash_cytoscape as cyto
import networkx as nx


app = dash.Dash(__name__)


def plot_graph_cytoscape(G: nx.Graph) -> None:
    """Plot a network graph using Cytoscape.

    Parameters
    ----------
    G : nx.Graph
        A NetworkX graph object. The nodes should be labeled with integers. The graph
        should be undirected.

    Returns
    -------
    None
    """
    nodes = [{"data": {"id": str(i), "label": f"Feature {i}"}} for i in G.nodes()]
    edges = [{"data": {"source": str(u), "target": str(v)}} for u, v in G.edges()]

    elements = nodes + edges

    app.layout = html.Div(
        [
            cyto.Cytoscape(
                id="network-graph",
                elements=elements,
                style={"width": "100%", "height": "800px"},
                layout={"name": "circle"},
                stylesheet=[
                    {
                        "selector": "node",
                        "style": {
                            "label": "data(label)",
                            "width": "20px",
                            "height": "20px",
                            "background-color": "#0074D9",
                            "color": "white",
                            "font-size": "10px",
                        },
                    },
                    {"selector": "edge", "style": {"line-color": "#B0BEC5"}},
                ],
            )
        ]
    )


if __name__ == "__main__":
    G = nx.complete_graph(10)
    plot_graph_cytoscape(G)
    app.run_server(debug=True)
