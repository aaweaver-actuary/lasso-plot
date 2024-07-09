"""Graph component."""

from __future__ import annotations
from dash import html
from app.components.feature_dropdown import FeatureDropdown


def Graph(features: list[str]) -> html.Div:
    """Create a graph component."""
    return (
        html.Div(
            children=[
                # html.Div(
                #     [
                #         html.Div(
                #             children=[
                #                 html.Label("Select a feature:")
                #                 # FeatureDropdown(
                #                 #     id="feature-dropdown", features=features
                #                 # ),
                #             ],
                #             className="y-axis-container",
                #         ),
                #         html.Div(children=[], className="graph-container")
                #     ]
                # ),
                # html.Div(
                #     children=[
                #         html.Div(children=[], className="blank-container"),
                #         html.Div(
                #             children=[
                #                 html.Label("Select a feature:")
                #                 # FeatureDropdown(
                #                 #     id="x-axis-dropdown", features=features
                #                 # ),
                #             ],
                #             className="x-axis-container",
                #         )
                #     ]
                # ),
            ],
            className="graph-container",
            style={"display": "flex"},
        ),
    )
