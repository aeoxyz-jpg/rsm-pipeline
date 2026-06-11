"""Common base class for column-keep-or-drop selectors."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class _SelectorBase(BaseEstimator, TransformerMixin):
    """Each subclass implements ``_fit_kept(X, y) -> list[str]``.

    The base class handles input validation, kept/dropped tracking, and
    DataFrame transform.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> "_SelectorBase":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("input must be a pandas DataFrame")
        self.feature_names_in_ = list(X.columns)
        kept_set = set(self._fit_kept(X, y))
        self.kept_columns_ = [c for c in self.feature_names_in_ if c in kept_set]
        self.dropped_columns_ = [c for c in self.feature_names_in_ if c not in kept_set]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("input must be a pandas DataFrame")
        return X[self.kept_columns_].copy()

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        return list(self.kept_columns_)

    def _fit_kept(self, X: pd.DataFrame, y: Any) -> list[str]:
        raise NotImplementedError

    def drop_reasons(self) -> dict[str, str]:
        return {c: self.__class__.__name__ for c in self.dropped_columns_}


class _PassthroughSelector(_SelectorBase):
    """No-op selector used when the selectors list is empty."""

    def _fit_kept(self, X: pd.DataFrame, y: Any) -> list[str]:
        return list(X.columns)

    def drop_reasons(self) -> dict[str, str]:
        return {}
