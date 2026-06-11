"""Smoothed mean target encoder (in-house, ~30 lines)."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TargetEncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing: float = 0.0):
        self.smoothing = smoothing

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TargetEncoderWrapper":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("input must be a pandas DataFrame")
        if y is None:
            raise ValueError("TargetEncoderWrapper.fit requires y")
        col = X.columns[0]
        s = X[col]
        global_mean = float(y.mean())
        encodings: dict[Any, float] = {}
        cat_count = s.value_counts(dropna=False)
        for cat, cnt in cat_count.items():
            mask = s.isna() if (cat is None or pd.isna(cat)) else s == cat
            cat_mean = float(y[mask].mean()) if mask.any() else global_mean
            denom = cnt + self.smoothing
            if denom <= 0:
                encodings[cat] = global_mean
            else:
                encodings[cat] = (cnt * cat_mean + self.smoothing * global_mean) / denom
        self.encodings_ = encodings
        self.global_mean_ = global_mean
        self.feature_names_in_ = [col]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("input must be a pandas DataFrame")
        col = X.columns[0]
        s = X[col]
        out = s.map(self.encodings_).fillna(self.global_mean_)
        return pd.DataFrame({col: out.astype("float64")}, index=X.index)

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        return list(self.feature_names_in_)
