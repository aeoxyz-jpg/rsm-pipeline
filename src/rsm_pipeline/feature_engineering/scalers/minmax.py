"""MinMaxScaler wrapper preserving DataFrame structure."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler


class MinMaxScalerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, low: float = 0.0, high: float = 1.0):
        self.low = low
        self.high = high

    def fit(self, X: pd.DataFrame, y: Any = None) -> "MinMaxScalerWrapper":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("input must be a pandas DataFrame")
        self.feature_names_in_ = list(X.columns)
        self._impl = MinMaxScaler(feature_range=(self.low, self.high))
        self._impl.fit(X)
        self.data_min_ = self._impl.data_min_
        self.data_max_ = self._impl.data_max_
        self.scale_ = self._impl.scale_
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("input must be a pandas DataFrame")
        arr = self._impl.transform(X)
        return pd.DataFrame(arr, columns=self.feature_names_in_, index=X.index)

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        return list(self.feature_names_in_)
