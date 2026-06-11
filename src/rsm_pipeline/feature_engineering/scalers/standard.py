"""StandardScaler wrapper preserving DataFrame structure."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class StandardScalerWrapper(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: Any = None) -> "StandardScalerWrapper":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("input must be a pandas DataFrame")
        self.feature_names_in_ = list(X.columns)
        self._impl = StandardScaler()
        self._impl.fit(X)
        self.mean_ = self._impl.mean_
        self.scale_ = self._impl.scale_
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("input must be a pandas DataFrame")
        arr = self._impl.transform(X)
        return pd.DataFrame(arr, columns=self.feature_names_in_, index=X.index)

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        return list(self.feature_names_in_)
