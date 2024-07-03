"""Model class to calculate the LASSO path for a logistic regression model."""

from __future__ import annotations
from sklearn.linear_model import LogisticRegressionCV
from dataclasses import dataclass
from typing import List

import pandas as pd
import numpy as np

from lasso_plot.interfaces import AbstractModel

__all__ = ["LassoLogisticModel"]


@dataclass
class LassoLogisticModel(AbstractModel):
    """Class to calculate the LASSO path for a logistic regression model.

    Attributes
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data.

    y : array-like of shape (n_samples,)
        The target values.

    n_alphas : int
        Number of alphas along the regularization path.

    alphas : array of shape (n_alphas,)
        The alphas along the regularization path.

    coefs : array of shape (n_features, n_alphas)
        Coefficients along the regularization path.
    """

    _n_alphas: int = 100
    _alphas: List[float] | None = None
    _coefs: List[float] | None = None
    _is_fit: bool = False

    @property
    def n_alphas(self) -> int:
        """Get the number of alphas along the regularization path."""
        return self._n_alphas

    @property
    def alphas(self) -> List[float]:
        """Get the alphas along the regularization path."""
        if self._alphas is None:
            raise ValueError("Model is not fit yet.")

        return self._alphas

    @property
    def coefs(self) -> List[float]:
        """Get the coefficients along the regularization path."""
        if self._coefs is None:
            raise ValueError("Model is not fit yet.")

        return self._coefs

    @property
    def is_fit(self) -> bool:
        """Check if the model is fit."""
        return self._is_fit

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> None:
        """Fit the logistic regression model with L1 regularization.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        y : array-like of shape (n_samples,)
            The target values.
        """
        clf = LogisticRegressionCV(
            Cs=self.n_alphas, cv=5, penalty="l1", solver="liblinear", random_state=0
        )
        clf.fit(X, y)

        self.alphas = 1 / clf.Cs_
        self.coefs = clf.coefs_paths_[1].mean(axis=0)

        self.is_fit = True

    def get_path(self) -> tuple:
        """Get the alphas and coefficients path.

        Returns
        -------
        alphas : array of shape (n_alphas,)
            The alphas along the regularization path.

        coefs : array of shape (n_features, n_alphas)
            The coefficients along the regularization path.
        """
        if not self.is_fit:
            raise ValueError("Model is not fit yet.")

        return self.alphas, self.coefs
