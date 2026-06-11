"""RobustScaler wrapper preserving DataFrame structure."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler


class RobustScalerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, q_low: float = 0.25, q_high: float = 0.75):
        self.q_low = q_low
        self.q_high = q_high

    def fit(self, X: pd.DataFrame, y: Any = None) -> "RobustScalerWrapper":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("input must be a pandas DataFrame")
        self.feature_names_in_ = list(X.columns)
        # sklearn expects percentile (0..100), our config carries fractions.
        self._impl = RobustScaler(quantile_range=(self.q_low * 100, self.q_high * 100))
        self._impl.fit(X)
        self.center_ = self._impl.center_
        self.scale_ = self._impl.scale_
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("input must be a pandas DataFrame")
        arr = self._impl.transform(X)
        return pd.DataFrame(arr, columns=self.feature_names_in_, index=X.index)

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        return list(self.feature_names_in_)
