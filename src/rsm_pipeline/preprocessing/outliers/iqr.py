"""IQR-based clipper: lower = Q1 - factor*IQR, upper = Q3 + factor*IQR."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class IQRClipper(BaseEstimator, TransformerMixin):
    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X: pd.DataFrame, y: Any = None) -> "IQRClipper":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("input must be a pandas DataFrame")
        self.feature_names_in_ = list(X.columns)
        q1 = np.nanquantile(X.to_numpy(dtype=float), 0.25, axis=0)
        q3 = np.nanquantile(X.to_numpy(dtype=float), 0.75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("input must be a pandas DataFrame")
        bounds_low = pd.Series(self.lower_, index=self.feature_names_in_)
        bounds_high = pd.Series(self.upper_, index=self.feature_names_in_)
        return X.clip(lower=bounds_low, upper=bounds_high, axis=1)

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        return list(self.feature_names_in_)
