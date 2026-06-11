"""Ordinal encoder wrapper.

Wraps ``sklearn.preprocessing.OrdinalEncoder(handle_unknown=..., unknown_value=...)``.
OOV at val/test maps to ``unknown_value`` (default -1).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder


class OrdinalEncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        handle_unknown: str = "use_encoded_value",
        unknown_value: int = -1,
    ):
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value

    def fit(self, X: pd.DataFrame, y: Any = None) -> "OrdinalEncoderWrapper":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("input must be a pandas DataFrame")
        self.feature_names_in_ = list(X.columns)
        kwargs: dict[str, Any] = {"handle_unknown": self.handle_unknown}
        if self.handle_unknown == "use_encoded_value":
            kwargs["unknown_value"] = self.unknown_value
        self._impl = OrdinalEncoder(**kwargs)
        self._impl.fit(X)
        self.categories_ = self._impl.categories_
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("input must be a pandas DataFrame")
        arr = self._impl.transform(X)
        return pd.DataFrame(
            arr.astype("int64"),
            columns=self.feature_names_in_,
            index=X.index,
        )

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        return list(self.feature_names_in_)
