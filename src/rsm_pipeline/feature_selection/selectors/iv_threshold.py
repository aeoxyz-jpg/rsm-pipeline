"""IV-threshold-based column drop with self-contained quick IV computation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rsm_pipeline.feature_selection.selectors._base import _SelectorBase


def _compute_iv(
    series: pd.Series,
    y: pd.Series,
    n_bins: int = 10,
    smoothing: float = 0.5,
) -> float:
    """Quick quantile-binning IV. Numeric-only; assumes input is already float."""
    arr = series.to_numpy(dtype=float)
    nan_mask = np.isnan(arr)
    if nan_mask.all():
        return 0.0
    try:
        cuts = np.unique(np.nanquantile(arr, np.linspace(0, 1, n_bins + 1)))
    except Exception:
        return 0.0
    if len(cuts) < 3:
        return 0.0

    bin_idx = np.digitize(arr, cuts[1:-1])
    bin_idx = np.where(nan_mask, len(cuts) - 1, bin_idx)
    y_arr = y.to_numpy()
    pos_mask = y_arr == 1
    neg_mask = y_arr == 0
    N_pos = float(pos_mask.sum())
    N_neg = float(neg_mask.sum())
    if N_pos == 0 or N_neg == 0:
        return 0.0

    unique_bins = np.unique(bin_idx)
    k = len(unique_bins)
    iv = 0.0
    for b in unique_bins:
        m = bin_idx == b
        n_pos = float((y_arr[m] == 1).sum())
        n_neg = float((y_arr[m] == 0).sum())
        pct_pos = (n_pos + smoothing) / (N_pos + smoothing * k)
        pct_neg = (n_neg + smoothing) / (N_neg + smoothing * k)
        woe = np.log(pct_pos / pct_neg)
        iv += (pct_pos - pct_neg) * woe
    return float(iv)


class IVThresholdSelector(_SelectorBase):
    def __init__(
        self,
        threshold: float = 0.02,
        n_bins: int = 10,
        smoothing: float = 0.5,
    ):
        self.threshold = threshold
        self.n_bins = n_bins
        self.smoothing = smoothing

    def _fit_kept(self, X: pd.DataFrame, y: Any) -> list[str]:
        if y is None:
            raise ValueError("IVThresholdSelector.fit requires y")
        y_series = pd.Series(y, index=X.index) if not isinstance(y, pd.Series) else y
        ivs: dict[str, float] = {}
        for col in X.columns:
            ivs[col] = _compute_iv(
                X[col], y_series, n_bins=self.n_bins, smoothing=self.smoothing
            )
        self.ivs_ = ivs
        return [c for c, iv in ivs.items() if iv >= self.threshold]

    def drop_reasons(self) -> dict[str, str]:
        dropped = set(self.dropped_columns_)
        return {
            col: f"iv={iv:.4f} < threshold={self.threshold}"
            for col, iv in self.ivs_.items()
            if col in dropped
        }
