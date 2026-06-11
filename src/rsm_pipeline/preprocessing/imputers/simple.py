"""Mean / Median / Mode imputers + ``_AppendIsMissing`` flag generator.

All transformers accept and return ``pd.DataFrame`` so column names
flow correctly through ``ColumnTransformer.get_feature_names_out``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


class _SimpleStrategyImputer(BaseEstimator, TransformerMixin):
    """Common base for mean / median / mode wrappers."""

    _strategy: str  # set by subclasses

    def fit(self, X: pd.DataFrame, y: Any = None) -> "_SimpleStrategyImputer":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("input must be a pandas DataFrame")
        self.feature_names_in_ = list(X.columns)
        self._impl = SimpleImputer(strategy=self._strategy)
        # Replace Python None with np.nan so sklearn recognises missing values
        # uniformly across object-dtype columns.
        self._impl.fit(X.where(X.notna(), other=np.nan))
        self.statistics_ = self._impl.statistics_
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("input must be a pandas DataFrame")
        arr = self._impl.transform(X.where(X.notna(), other=np.nan))
        return pd.DataFrame(arr, columns=self.feature_names_in_, index=X.index)

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        return list(self.feature_names_in_)


class MeanImputer(_SimpleStrategyImputer):
    """Numeric only — mean imputation."""

    _strategy = "mean"


class MedianImputer(_SimpleStrategyImputer):
    """Numeric only — median imputation."""

    _strategy = "median"


class ModeImputer(_SimpleStrategyImputer):
    """Categorical only — most_frequent imputation."""

    _strategy = "most_frequent"


class _AppendIsMissing(BaseEstimator, TransformerMixin):
    """For each input column ``c``, append ``c+'_is_missing'`` (1 if NaN else 0).

    The original columns are passed through unchanged; the downstream
    imputer handles the actual NaN replacement. Indicator columns are
    created at ``fit`` time (column-set frozen there) so val/test always
    receive the same columns even if their NaN patterns differ from train.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> "_AppendIsMissing":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("input must be a pandas DataFrame")
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("input must be a pandas DataFrame")
        X = X.copy()
        flags = X.isna().astype("int8").add_suffix("_is_missing")
        return pd.concat([X, flags], axis=1)

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        cols = (
            list(input_features)
            if input_features is not None
            else list(self.feature_names_in_)
        )
        return cols + [f"{c}_is_missing" for c in cols]
