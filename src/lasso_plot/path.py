"""Define objects representing a LASSO path and a LASSO path plotter."""

from __future__ import annotations
from dataclasses import dataclass
from lasso_plot.interfaces import AbstractModel, AbstractPath, AbstractPlotter
from typing import List

import plotly.graph_objects as go

__all__ = ["LassoPath", "LassoPathPlotter"]

@dataclass
class LassoPath(AbstractPath):
    """Class representing a single LASSO path."""

    alphas: List[float]
    coefs: List[float]
    name: str
    model: AbstractModel

    def get_path(self) -> tuple:
        """Get the alphas and coefficients for a single LASSO path."""
        return self.model.get_path()

@dataclass
class LassoPathPlotter(AbstractPlotter):
    """Class responsible for plotting a LASSO path."""

    def plot(self, path: AbstractPath, fig: go.Figure | None = None) -> go.Figure:
        """
        Plot the LASSO path for a logistic regression model.

        Parameters
        ----------
        path : AbstractPath
            The LASSO path to plot.
        fig : plotly.graph_objects.Figure
            The figure to plot the LASSO path on. If None, a new figure is created.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            The LASSO path plot.
        """
        if fig is None:
            fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=path.alphas,
                y=path.coefs,
                mode="lines",
                name=path.name,
                legendgroup="coefs",
            )
        )

        return fig