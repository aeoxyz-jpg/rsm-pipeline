"""KNN imputer wrapper.

Wraps ``sklearn.impute.KNNImputer``. Numeric only. Memory caveat: the
distance matrix is O(n_train²); avoid for n > 50k rows.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer


class KNNImputerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors

    def fit(self, X: pd.DataFrame, y: Any = None) -> "KNNImputerWrapper":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("input must be a pandas DataFrame")
        self.feature_names_in_ = list(X.columns)
        self._impl = KNNImputer(n_neighbors=self.n_neighbors)
        self._impl.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("input must be a pandas DataFrame")
        arr = self._impl.transform(X)
        return pd.DataFrame(arr, columns=self.feature_names_in_, index=X.index)

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        return list(self.feature_names_in_)
