"""Z-score capper: lower / upper = mean ± threshold * std (ddof=0)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ZScoreCapper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: Any = None) -> "ZScoreCapper":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("input must be a pandas DataFrame")
        self.feature_names_in_ = list(X.columns)
        arr = X.to_numpy(dtype=float)
        self.mean_ = np.nanmean(arr, axis=0)
        self.std_ = np.nanstd(arr, axis=0, ddof=0)
        lower = self.mean_ - self.threshold * self.std_
        upper = self.mean_ + self.threshold * self.std_
        const_mask = self.std_ == 0
        lower = np.where(const_mask, -np.inf, lower)
        upper = np.where(const_mask, np.inf, upper)
        self.lower_ = lower
        self.upper_ = upper
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("input must be a pandas DataFrame")
        bounds_low = pd.Series(self.lower_, index=self.feature_names_in_)
        bounds_high = pd.Series(self.upper_, index=self.feature_names_in_)
        return X.clip(lower=bounds_low, upper=bounds_high, axis=1)

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        return list(self.feature_names_in_)
