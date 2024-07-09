"""Protocol for the input data."""

from __future__ import annotations
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import pandas as pd


@dataclass
class Data:
    """Data class to store the input data."""

    _X: pd.DataFrame | None = None
    _y: pd.Series | None = None

    def __post_init__(self):
        """Generate the input data."""
        _X, _y = make_classification(n_samples=1000, n_features=20, random_state=42)
        self._X = pd.DataFrame(
            _X,
            columns=[
                f"feature_{'0' if i+1 < 10 else ''}{i+1}" for i in range(_X.shape[1])
            ],
        )
        self._y = pd.Series(_y, name="target")

    @property
    def split(
        self,
    ) -> tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
    ]:
        """Split the input data into training and testing sets."""
        X_train, X_test, y_train, y_test = train_test_split(
            self._X, self._y, random_state=42, test_size=0.2
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, random_state=42, test_size=0.25
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    @property
    def features(self) -> list[str]:
        """Return the feature names."""
        return self._X.columns.tolist()

    @property
    def target(self) -> str:
        """Return the target name."""
        return self._y.name

    @property
    def train(self) -> tuple[pd.DataFrame, pd.Series]:
        """Return the training data."""
        X_train, _, _, y_train, _, _ = self.split
        return X_train, y_train

    @property
    def val(self) -> tuple[pd.DataFrame, pd.Series]:
        """Return the validation data."""
        _, X_val, _, _, y_val, _ = self.split
        return X_val, y_val

    @property
    def test(self) -> tuple[pd.DataFrame, pd.Series]:
        """Return the testing data."""
        _, _, X_test, _, _, y_test = self.split
        return X_test, y_test

    @property
    def X(self) -> pd.DataFrame:
        """Return the input data."""
        return self._X

    @property
    def y(self) -> pd.Series:
        """Return the target data."""
        return self._y

    @property
    def X_train(self) -> pd.DataFrame:
        """Return the training input data."""
        return self.train[0]

    @property
    def y_train(self) -> pd.Series:
        """Return the training target data."""
        return self.train[1]

    @property
    def X_val(self) -> pd.DataFrame:
        """Return the validation input data."""
        return self.val[0]

    @property
    def y_val(self) -> pd.Series:
        """Return the validation target data."""
        return self.val[1]

    @property
    def X_test(self) -> pd.DataFrame:
        """Return the testing input data."""
        return self.test[0]

    @property
    def y_test(self) -> pd.Series:
        """Return the testing target data."""
        return self.test[1]

    def __repr__(self) -> str:
        """Return the string representation of the class."""
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        """Return the string representation of the class."""
        return f"{self.__class__.__name__}()"


__all__ = ["Data"]
