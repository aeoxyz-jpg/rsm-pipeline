"""NumericalWoEEncoder — fit cuts via quantile or monotonic, output WoE values."""

from __future__ import annotations

from math import log
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from rsm_pipeline.feature_engineering.woe.binning import (
    monotonic_bins,
    quantile_bins,
)
from rsm_pipeline.feature_engineering.woe.iv import WoEBin, WoETable


def _format_label(lo: float, hi: float) -> str:
    return f"({lo}, {hi}]"


class NumericalWoEEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        binning: str = "quantile",
        n_bins: int = 10,
        min_bin_pct: float = 0.05,
    ):
        self.binning = binning
        self.n_bins = n_bins
        self.min_bin_pct = min_bin_pct

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NumericalWoEEncoder":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("input must be a pandas DataFrame")
        if y is None:
            raise ValueError("NumericalWoEEncoder.fit requires y")
        col = X.columns[0]
        s = X[col]

        if self.binning == "quantile":
            cuts = quantile_bins(s, self.n_bins)
        elif self.binning == "monotonic":
            cuts = monotonic_bins(s, y, self.n_bins, self.min_bin_pct)
        else:
            raise ValueError(f"unknown binning: {self.binning!r}")

        # Bin assignment: np.digitize(values, cuts[1:-1]) -> 0..len(cuts)-2.
        nan_mask = s.isna().to_numpy()
        values = s.to_numpy()
        # Replace NaN with a finite placeholder for digitize (those rows handled separately).
        values_filled = np.where(nan_mask, 0.0, values)
        bin_idx = np.digitize(values_filled, np.asarray(cuts[1:-1]))

        n_bins = len(cuts) - 1
        n_pos_total_no_nan = 0
        n_neg_total_no_nan = 0
        bin_counts: list[tuple[int, int]] = []
        y_arr = y.to_numpy()
        for i in range(n_bins):
            mask = (bin_idx == i) & (~nan_mask)
            n_pos = int(y_arr[mask].sum())
            n_neg = int(mask.sum() - n_pos)
            bin_counts.append((n_pos, n_neg))
            n_pos_total_no_nan += n_pos
            n_neg_total_no_nan += n_neg

        nan_pos = int(y_arr[nan_mask].sum())
        nan_neg = int(nan_mask.sum() - nan_pos)
        has_nan = bool(nan_mask.any())

        n_pos_total = n_pos_total_no_nan + nan_pos
        n_neg_total = n_neg_total_no_nan + nan_neg
        k = n_bins + (1 if has_nan else 0)

        def woe_iv(n_pos: int, n_neg: int) -> tuple[float, float]:
            pct_pos = (n_pos + 0.5) / (n_pos_total + 0.5 * k)
            pct_neg = (n_neg + 0.5) / (n_neg_total + 0.5 * k)
            woe = log(pct_pos / pct_neg)
            iv_contrib = (pct_pos - pct_neg) * woe
            return woe, iv_contrib

        bins: list[WoEBin] = []
        bin_woe = np.zeros(n_bins, dtype="float64")
        for i, (n_pos, n_neg) in enumerate(bin_counts):
            n_total = n_pos + n_neg
            pos_rate = (n_pos / n_total) if n_total > 0 else 0.0
            woe, ivc = woe_iv(n_pos, n_neg)
            bins.append(
                WoEBin(
                    label=_format_label(cuts[i], cuts[i + 1]),
                    n_total=n_total,
                    n_pos=n_pos,
                    n_neg=n_neg,
                    pos_rate=pos_rate,
                    woe=woe,
                    iv_contrib=ivc,
                )
            )
            bin_woe[i] = woe

        nan_woe_value = 0.0
        if has_nan:
            woe_n, ivc_n = woe_iv(nan_pos, nan_neg)
            nan_woe_value = woe_n
            n_total_n = nan_pos + nan_neg
            bins.append(
                WoEBin(
                    label="__nan__",
                    n_total=n_total_n,
                    n_pos=nan_pos,
                    n_neg=nan_neg,
                    pos_rate=(nan_pos / n_total_n) if n_total_n > 0 else 0.0,
                    woe=woe_n,
                    iv_contrib=ivc_n,
                )
            )

        iv = float(sum(b.iv_contrib for b in bins))

        self.cuts_ = list(cuts)
        self.bin_woe_ = bin_woe
        self.nan_woe_ = float(nan_woe_value)
        self.table_ = WoETable(
            column=col,
            kind="numerical",
            bins=bins,
            iv=iv,
            n_pos_total=n_pos_total,
            n_neg_total=n_neg_total,
            nan_woe=float(nan_woe_value),
        )
        self.feature_names_in_ = [col]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("input must be a pandas DataFrame")
        col = X.columns[0]
        s = X[col]
        nan_mask = s.isna().to_numpy()
        values = s.to_numpy()
        values_filled = np.where(nan_mask, 0.0, values)
        bin_idx = np.digitize(values_filled, np.asarray(self.cuts_[1:-1]))
        out = self.bin_woe_[bin_idx]
        out = np.where(nan_mask, self.nan_woe_, out)
        return pd.DataFrame({col: out.astype("float64")}, index=X.index)

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        return list(self.feature_names_in_)
