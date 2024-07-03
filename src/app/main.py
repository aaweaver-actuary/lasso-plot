"""Create a Dash app to visualize the results."""

import dash  # type: ignore
from dash import dcc, html, Input, Output
import plotly.express as px  # type: ignore
import pandas as pd

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div(
    [
        html.H1("Interactive Dash App"),
        dcc.Dropdown(
            id="city-dropdown",
            options=[{"label": city, "value": city} for city in df["City"].unique()],
            value="SF",
        ),
        dcc.Graph(id="bar-chart"),
    ]
)


# Callback to update the bar chart based on the selected city
@app.callback(Output("bar-chart", "figure"), Input("city-dropdown", "value"))
def update_chart(selected_city):
    filtered_df = df[df["City"] == selected_city]
    fig = px.bar(filtered_df, x="Fruit", y="Amount", title=f"Fruits in {selected_city}")
    return fig


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
