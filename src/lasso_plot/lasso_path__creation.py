"""Create a LASSO path plot for the top N features."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import plotly.graph_objects as go
import polars as pl
from predictables import DuckDB

DATABASE_FILE = "./hit_ratio_results.duckdb"
TOP_N_FEATURES = 25
N_ALPHA = 1000
HTML_OUTPUT_FILE = f"./lasso_plot_top_{TOP_N_FEATURES}.html"

DataFrameLike = pl.DataFrame | pl.LazyFrame


def get_most_recent_run(db: DuckDB) -> str:
    """Return the most recent run_id."""
    return db("""
    with
    
    -- most recent run timestamp
    max_run_at as (
        select max(run_at) as max_run_at from lasso_plot.run_lookup 
    ),
    
    -- run_id corresponding to timestamp
    lookup as (
        select distinct rl.run_id
        from lasso_plot.run_lookup as rl, max_run_at
        where run_at = max_run_at
    )
    
    from lookup
    """).item(0, 0)


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
        x=[0, N_ALPHA], y=[0, 0], name=name, line={"color": lc, "width": lw, "dash": ls}
    )

    return fig


def handle_window_size(results: DataFrameLike) -> Tuple[float, float]:
    """Adjust the window size to always include the y=0 line, while ensuring that outliers on the y-axis do not distort the plot."""
    # Get the first and 99th quantiles of the data, excluding the alpha column and all 0 values
    arr = (
        results.select([c for c in results.columns if c not in ["alpha", "run_id"]])
        .collect()
        .to_numpy()
        .flatten()
    )
    min_y, max_y = np.quantile(arr[arr != 0], [0.01, 0.99])

    # Return the min and max values
    return min(min_y, 0), max(max_y, 0)


def __top_features_l2_strategy(
    results: DataFrameLike, n: int = TOP_N_FEATURES
) -> pl.DataFrame:
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
    results: DataFrameLike, n: int = TOP_N_FEATURES
) -> pl.DataFrame:
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
    results: DataFrameLike, n: int = TOP_N_FEATURES, strategy: str = "alpha_rank"
) -> pl.DataFrame:
    """Sort the features based on their importance to the LASSO regression model."""
    if strategy == "l2_norm":
        return __top_features_l2_strategy(results, n)
    elif strategy == "alpha_rank":
        return __top_features_alpha_rank_strategy(results, n)
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


def update_y_axis(fig: go.Figure, results: DataFrameLike) -> go.Figure:
    """Update the y-axis label and range."""
    min_y, max_y = handle_window_size(results)
    fig.update_layout(
        yaxis={
            "title": "10-alpha Moving Average Fitted Coefficient",
            "range": [min_y, max_y],
        }
    )

    return fig


def update_title(fig: go.Figure, title: str | None = None) -> go.Figure:
    """Update the y-axis label and range."""
    title_ = (
        (
            "<b>LASSO Regularization Path - Top 50 Features</b><br>"
            "Impact on fitted coefficients as the L1-regularization penalty is slowly increased<br>"
            "Features are ordered by when this regularization penalty forces them to drop out of the model (eg their coefficient becomes 0)"
        )
        if title is None
        else title
    )

    fig.update_layout(title={"text": title_})

    return fig


def build_plot(results: DataFrameLike, db: DuckDB) -> None:
    """Build the LASSO path plot for the top N features, and save it to an HTML file."""
    fig = go.Figure()

    # Add all the individual paths
    for feature in top_features(results, TOP_N_FEATURES).tolist():
        fig = add_single_path(fig, feature, db)

    # Add some layout elements, update axes, etc
    fig = add_zero_line(fig)
    fig = update_x_axis(fig, db)
    fig = update_y_axis(fig, results)
    fig = update_title(fig)

    # Save the file
    fig.write_html(HTML_OUTPUT_FILE)


def main() -> None:
    """Build the LASSO path plot.

    1. Initialize the database.
    2. Get the most recent run_id.
    3. Get the results data.
    4. Build the plot.
    """
    # initialize the database
    db = DuckDB(DATABASE_FILE)

    # most recent run_id
    max_run_id = get_most_recent_run(db)

    # get the results data
    results = db(f"from lasso_plot.result where run_id='{max_run_id}'").lazy()

    # build the plot
    build_plot(results, db)


if __name__ == "__main__":
    main()
