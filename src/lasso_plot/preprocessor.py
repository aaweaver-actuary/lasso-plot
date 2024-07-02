"""Preprocess the data for the LASSO path plot."""

from __future__ import annotations
import pandas as pd
from sklearn.preprocessing import StandardScaler

from lasso_plot.interfaces import AbstractPreprocessor

from dataclasses import dataclass

__all__ = ["Preprocessor"]


@dataclass
class Preprocessor(AbstractPreprocessor):
    """Class for standard scaling preprocessing."""

    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        """Standardize the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        X_standardized : array-like of shape (n_samples, n_features)
            The standardized data.
        """
        arr = StandardScaler().fit_transform(X)
        return pd.DataFrame(arr, columns=X.columns)
