"""Create a Dash app to visualize the results."""

from __future__ import annotations
from dash import html, dcc  # type: ignore
from app.Layout import Layout
from app.components.SelectedSeriesScatterplot import SelectedSeriesScatterplot
from app.components.base import h1
from app._app_config import app, data


# Layout of the app
app.layout = Layout(
    [
        h1("Scatterplot"),
        dcc.Store(
            id="selected-features-store",
            data={"x": data.features[0], "y": data.features[1]},
        ),
        html.Div(
            [
                html.Div(
                    [
                        SelectedSeriesScatterplot(
                            xaxis_column_name=None, yaxis_column_name=None
                        )
                    ],
                    id="scatterplot-container",
                    style={
                        "border": "1px solid black",
                        "margin": "5px",
                        "height": "87%",
                    },
                ),
                html.Div(
                    [html.Div()],
                    className="x-axis-container",
                    style={
                        "border": "1px solid black",
                        "margin": "5px",
                        "height": "10%",
                    },
                ),
            ],
            className="flex flex-col border-[1px] border-black h-[79vh] m-[10px] w-full",
        ),
    ]
)


if __name__ == "__main__":
    app.run(debug=True)
