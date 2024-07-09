"""Create a scatterplot of the selected series."""

from __future__ import annotations
from dash import html, Input, Output
from app._app_config import app, eids, data as _data

from app.components.base import h5, Dropdown, Scatterplot

eids.add("eda_scatterplot", ["x", "y", "axes_picker"])


@app.callback(
    Output(eids.eda_scatterplot.container, "children"),
    Input(eids.eda_scatterplot.x, "value"),
    Input(eids.eda_scatterplot.y, "value"),
)
def SelectedSeriesScatterplot(
    xaxis_column_name: str | None = None, yaxis_column_name: str | None = None
) -> html.Div:
    """Create a scatterplot of the selected series."""
    if xaxis_column_name is None:
        xaxis_column_name = _data.features[0]
    if yaxis_column_name is None:
        yaxis_column_name = _data.features[1]

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            h5("Select a feature for the x-axis:"),
                            Dropdown(
                                id=eids.eda_scatterplot.x,
                                features=_data.features,
                                placeholder=_data.features[0],
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            h5("Select a feature for the y-axis:"),
                            Dropdown(
                                id=eids.eda_scatterplot.y,
                                features=_data.features,
                                placeholder=_data.features[1],
                            ),
                        ]
                    ),
                ],
                id=eids.eda_scatterplot.axes_picker,
            ),
            Scatterplot(
                x1=_data.X_train[xaxis_column_name],
                x2=_data.X_train[yaxis_column_name],
                y=_data.y_train,
            ),
        ],
        id=eids.eda_scatterplot.container,
    )
