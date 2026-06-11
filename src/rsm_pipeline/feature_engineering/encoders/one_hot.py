"""OneHot encoder wrapper.

Wraps ``sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore', sparse_output=False)``
so OOV categories at val/test produce all-zero rows rather than errors.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class OneHotEncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, drop: Optional[str] = None):
        self.drop = drop

    def fit(self, X: pd.DataFrame, y: Any = None) -> "OneHotEncoderWrapper":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("input must be a pandas DataFrame")
        self.feature_names_in_ = list(X.columns)
        self._impl = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            drop=self.drop,
        )
        self._impl.fit(X)
        self.categories_ = self._impl.categories_
        self._out_names = list(self._impl.get_feature_names_out(self.feature_names_in_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("input must be a pandas DataFrame")
        arr = self._impl.transform(X)
        return pd.DataFrame(arr, columns=self._out_names, index=X.index)

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        return list(self._out_names)
