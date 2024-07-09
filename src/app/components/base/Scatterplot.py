"""Base scatterplot component."""

from app._types import ArrayLike
from dash import html, dcc
import plotly.graph_objects as go


def target_figure(
    x1: ArrayLike, x2: ArrayLike, y: ArrayLike, target_value: int
) -> go.Scatter:
    """Create a scatter plot of a target value."""
    return go.Scatter(
        x=x1[y == target_value],
        y=x2[y == target_value],
        mode="markers",
        marker={"color": "skyblue" if target_value == 0 else "darkred"},
        name=str(target_value),
        legendgroup="target",
        legendgrouptitle={"text": "Target"},
    )


def Scatterplot(x1: ArrayLike, x2: ArrayLike, y: ArrayLike) -> html.Div:
    """Create a scatterplot of the selected series."""
    fig = go.Figure()
    fig.add_trace(target_figure(x1, x2, y, 0))
    fig.add_trace(target_figure(x1, x2, y, 1))

    return html.Div(
        [dcc.Graph(figure=fig, className="scatterplot")],
        className="scatterplot-container",
    )
