"""Plot the LASSO regularization path as a feature selection tool."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from predictables import DuckDB

DATABASE_FILE = "./hit_ratio_results.duckdb"
TOP_N_FEATURES = 50
HTML_OUTPUT_FILE = f"./lasso_plot_top_{TOP_N_FEATURES}.html"


def add_single_path(fig: go.Figure, feature: str, db: DuckDB) -> go.Figure:
    """Add a path for a single variable to the figure and return the updated figure."""
    data = db(f"""
        select
            alpha,
            avg(\"{feature}\") over (rows between 4 preceding and 4 following) as \"{feature}\"

        from lasso_plot.result
        where alpha > 0
        order by alpha
    """)
    fig.add_scatter(
        x=data["alpha"].to_numpy(),
        y=data[feature].to_numpy(),
        name=feature,
        mode="lines",
    )

    return fig


def add_zero_line(
    fig: go.Figure, name: str = "y=0", lc: str = "black", lw: int = 4, ls: str = "dash"
) -> go.Figure:
    """Add a horizontal line at y=0 to indicate when the feature has dropped out of the model."""
    fig.add_scatter(
        x=[0, 100], y=[0, 0], name=name, line={"color": lc, "width": lw, "dash": ls}
    )

    return fig


def handle_window_size(
    results: pl.LazyFrame | pl.DataFrame, db: DuckDB
) -> Tuple[float, float]:
    """Adjust the window size to always include the y=0 line."""
    data_y = (
        db("""
        from lasso_plot.result
        where alpha > 0
        order by alpha
    """)
        .lazy()
        .select(["alpha", *top_features(results, TOP_N_FEATURES).tolist()])
        .drop("alpha")
        .collect()
        .to_numpy()
        .flatten()
    )

    return min(np.min(data_y), 0), max(np.max(data_y), 0)


def __top_features_l2_strategy(
    results: pl.LazyFrame | pl.DataFrame, n: int = TOP_N_FEATURES
) -> pd.Series:
    """Separate the l2 strategy from the logic of which strategy is chosen."""
    return (
        results.select(
            [
                pl.col(c).pow(2).sum().name.keep()
                for c in results.columns
                if c not in ["alpha"]
            ]
        )
        .collect()
        .transpose(
            include_header=True,
            header_name="feature",
            column_names=["l2_norm_lasso_coef"],
        )
        .sort("l2_norm_lasso_coef", descending=True)
        .head(n)
        .to_series()
        .to_pandas()
    )


def __top_features_alpha_rank_strategy(
    results: pl.LazyFrame | pl.DataFrame, n: int = TOP_N_FEATURES
) -> pd.Series:
    """Separate the alpha ranking strategy from the logic of which strategy is chosen."""
    return (
        results.sort("alpha")
        .filter(pl.col("alpha") > 0)
        .select(
            [
                pl.col("alpha").filter(pl.col(c) == 0).min().fill_null(99999).alias(c)
                for c in results.columns
                if c not in ["alpha", "run_id"]
            ]
        )
        .collect()
        .transpose(
            include_header=True,
            header_name="feature",
            column_names=["alpha_where_feature_drops_out"],
        )
        .sort("alpha_where_feature_drops_out", descending=True)
        .head(n)
        .to_series()
        .to_pandas()
    )


def top_features(
    results: pl.LazyFrame | pl.DataFrame, strategy: str = "alpha_rank"
) -> pd.Series:
    """Sort the features based on their importance to the LASSO regression model."""
    if strategy == "l2_norm":
        return __top_features_l2_strategy(results)
    elif strategy == "alpha_rank":
        return __top_features_alpha_rank_strategy(results)
    else:
        raise NotImplementedError(
            f"strategy {strategy} has not been implemented for top_features. Use `l2_norm` or `alpha_rank` instead."
        )


def update_x_axis(fig: go.Figure, db: DuckDB) -> go.Figure:
    """Update the x-axis label and range."""
    data = db("""
        select alpha
        from lasso_plot.result
        where alpha > 0
        order by alpha
    """).to_pandas()
    fig.update_layout(
        xaxis={"title": "L1 Regularization Strength", "range": [0, data["alpha"].max()]}
    )

    return fig


def update_y_axis(
    fig: go.Figure, results: pl.LazyFrame | pl.DataFrame, db: DuckDB
) -> go.Figure:
    """Update the y-axis label and range."""
    min_y, max_y = handle_window_size(results, db)
    fig.update_layout(
        yaxis={
            "title": "10-alpha Moving Average Fitted Coefficient",
            "range": [min_y, max_y],
        }
    )

    return fig


def update_title(fig: go.Figure, title: str | None = None) -> go.Figure:
    """Update the y-axis label and range."""
    title = (
        (
            "<b>LASSO Regularization Path - Top 50 Features</b><br>"
            "Impact on fitted coefficients as the L1-regularization penalty is slowly increased<br>"
            "Features are ordered by when this regularization penalty forces them to drop out of the model (eg their coefficient becomes 0)"
        )
        if title is None
        else title
    )

    fig.update_layout(title=title)

    return fig


def build_plot(results: pl.LazyFrame | pl.DataFrame, db: DuckDB) -> None:
    """Build the plot and save it to an HTML file."""
    fig = go.Figure()

    # Add all the individual paths
    for feature in top_features(results, TOP_N_FEATURES).tolist():
        fig = add_single_path(fig, feature, db)

    # Add some layout elements, update axes, etc
    fig = add_zero_line(fig)
    fig = update_x_axis(fig, db)
    fig = update_y_axis(fig, results, db)

    # Save the file
    fig.write_html(HTML_OUTPUT_FILE)


def main() -> None:
    """Define the main functionality of the script."""
    # initialize the database
    db = DuckDB(DATABASE_FILE)

    # get the results data
    results = db("from lasso_plot.result").lazy()

    # build the plot
    build_plot(results, db)


if __name__ == "__main__":
    main()
