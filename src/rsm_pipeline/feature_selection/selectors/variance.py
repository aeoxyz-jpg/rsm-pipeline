"""Variance-based column drop (wraps sklearn.VarianceThreshold)."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from rsm_pipeline.feature_selection.selectors._base import _SelectorBase


class VarianceSelector(_SelectorBase):
    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def _fit_kept(self, X: pd.DataFrame, y: Any) -> list[str]:
        impl = VarianceThreshold(threshold=self.threshold)
        impl.fit(X)
        self.variances_ = impl.variances_
        mask = impl.get_support()
        return [c for c, m in zip(X.columns, mask) if m]

    def drop_reasons(self) -> dict[str, str]:
        out: dict[str, str] = {}
        dropped = set(self.dropped_columns_)
        for col, var in zip(self.feature_names_in_, self.variances_):
            if col in dropped:
                out[col] = f"variance={float(var):.6f} <= threshold={self.threshold}"
        return out
