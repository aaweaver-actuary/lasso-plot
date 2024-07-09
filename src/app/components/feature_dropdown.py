"""Feature dropdown component."""

from __future__ import annotations
from dash import dcc, html


def FeatureDropdown(id: str, features: list[str]) -> html.Div:
    """Create a feature dropdown component."""
    return dcc.Dropdown(
        id=id,
        options=[{"label": i, "value": i} for i in features],
        value=features[0],
        style={"border-radius": "5px"},
    )
