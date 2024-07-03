"""Create the datasets used to build the LASSO regularization path plot."""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import polars.selectors as cs
from predictables import DuckDB, HitRatioFineTuningDB
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DB_FILE_PATH = "/sas/data/project/EG/ActShared/SmallBusiness/Modeling/hit_ratio/hit_ratio_results.duckdb"


@dataclass
class LassoPathData:
    """Data used to build the LASSO regularization path."""

    db: DuckDB | None = None
    model: BaseEstimator | None = None

    def __post_init__(self):
        """Initialize the database connection if it is not already set."""
        if self.db is None:
            self.db = HitRatioFineTuningDB()

        if self.model is None:
            self.modle = joblib.load(DB_FILE_PATH)

    @property
    def X(self) -> pd.DataFrame:
        """Return the feature matrix."""
        return (
            self.db("from train.X")
            .lazy()
            .select([*self.model.feature_names_, "hit_count"])
            .drop(["row_id", "hit_count"])
            .collect()
            .to_pandas()
            .set_index("quote_key")
        )

    @property
    def X_cat(self) -> pd.DataFrame:
        """Return the categorical features."""
        return pd.get_dummies(
            self.X.select([cs.categorical(), cs.string()]).collect().to_pandas()
        ).astype(int)

    @property
    def X_num(self) -> pd.DataFrame:
        """Return the numerical features."""
        return (
            self.X.drop([cs.categorical(), cs.string(), "hit_count"])
            .select(
                [
                    pl.when(pl.col(c).std() > 0)
                    .then((pl.col(c) - pl.col(c).mean()) / pl.col(c).std())
                    .otherwise(pl.col(c) - pl.col(c).mean())
                    .fill_nan(pl.col(c).drop_nans().min())
                    for c in self.X.drop(
                        [cs.categorical(), cs.string(), "hit_count"]
                    ).columns
                ]
            )
            .collect()
            .to_pandas()
        )

    @property
    def y(self) -> pd.Series:
        """Return the target vector."""
        return (
            self.db("from train.X")
            .lazy()
            .select([*self.model.feature_names_, "hit_count"])
            .select(["quote_key", "hit_count"])
            .collect()
            .to_pandas()
            .set_index("quote_key")
        )

    @property
    def X_processed(self) -> pd.DataFrame:
        """Return the processed feature matrix."""
        idx = self.y.index
        df = pd.concat([self.X_cat, self.X_num], axis=1)
        df.index = idx
        return df


def main():
    db = HitRatioFineTuningDB()
    # hr_model = joblib.load(DB_FILE_PATH)

    # X_cat = df.select(X.columns).select([cs.categorical(), cs.string()])
    # X_num = df.select(X.columns).drop([cs.categorical(), cs.string(), "hit_count"])

    # X_cat_processed = pd.get_dummies(X_cat.collect().to_pandas()).astype(int)
    # X_num_processed = (
    #     X_num.select(
    #         [
    #             pl.when(pl.col(c).std() > 0)
    #             .then((pl.col(c) - pl.col(c).mean()) / pl.col(c).std())
    #             .otherwise(pl.col(c) - pl.col(c).mean())
    #             .fill_nan(pl.col(c).drop_nans().min())
    #             for c in X_num.columns
    #         ]
    #     )
    #     .collect()
    #     .to_pandas()
    # )

    # X_processed = pd.concat([X_cat_processed, X_num_processed], axis=1)
    # X_processed.index = y.index

    df = (
        pd.concat([X_processed, y], axis=1)
        .dropna()
        .drop(columns=["log1p[acct_annual_revenue]", "log1p[cin_bop_hno_liab_prem]"])
    )

    x0 = df.drop(columns=["hit_count"])  # noqa: F841
    y0 = df["hit_count"]
    y0_ = y0.to_frame()  # noqa: F841

    db.write("create or replace table lasso_plot.X as select * from x0;")
    db.write("create or replace table lasso_plot.y as select * from y0_;")


if __name__ == "__main__":
    main()
