"""Percentile clipper: lower / upper = quantile(col, [low, high])."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PercentileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, low: float = 0.01, high: float = 0.99):
        self.low = low
        self.high = high

    def fit(self, X: pd.DataFrame, y: Any = None) -> "PercentileClipper":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("input must be a pandas DataFrame")
        self.feature_names_in_ = list(X.columns)
        arr = X.to_numpy(dtype=float)
        self.lower_ = np.nanquantile(arr, self.low, axis=0)
        self.upper_ = np.nanquantile(arr, self.high, axis=0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("input must be a pandas DataFrame")
        bounds_low = pd.Series(self.lower_, index=self.feature_names_in_)
        bounds_high = pd.Series(self.upper_, index=self.feature_names_in_)
        return X.clip(lower=bounds_low, upper=bounds_high, axis=1)

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        return list(self.feature_names_in_)
