"""Custom-fill imputer — replaces NaN with a user-specified scalar.

Stateless beyond the configured ``fill_value``. Works on numeric or
categorical columns; the fill_value must be assignable into the column's
dtype (e.g. ``-1`` for int, ``"unknown"`` for object).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CustomFillImputer(BaseEstimator, TransformerMixin):
    def __init__(self, fill_value: Any):
        self.fill_value = fill_value

    def fit(self, X: pd.DataFrame, y: Any = None) -> "CustomFillImputer":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("input must be a pandas DataFrame")
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("input must be a pandas DataFrame")
        return X.fillna(self.fill_value)

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        return list(self.feature_names_in_)
