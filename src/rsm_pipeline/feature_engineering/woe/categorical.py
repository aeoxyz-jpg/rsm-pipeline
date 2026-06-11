"""CategoricalWoEEncoder — rare-bucket merging + smoothed WoE."""

from __future__ import annotations

from math import log
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from rsm_pipeline.feature_engineering.woe.iv import WoEBin, WoETable


class CategoricalWoEEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, min_bin_pct: float = 0.01):
        self.min_bin_pct = min_bin_pct

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CategoricalWoEEncoder":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("input must be a pandas DataFrame")
        if y is None:
            raise ValueError("CategoricalWoEEncoder.fit requires y")

        col = X.columns[0]
        s = X[col]
        n = len(s)
        threshold = max(1, int(n * self.min_bin_pct))
        counts = s.value_counts(dropna=False)
        kept = [c for c, cnt in counts.items() if cnt >= threshold]
        rare = [c for c, cnt in counts.items() if cnt < threshold]

        y_arr = y.to_numpy()

        kept_records: list[tuple[Any, int, int]] = []  # (cat, n_pos, n_neg)
        for cat in kept:
            mask = (
                s.isna().to_numpy()
                if (cat is None or pd.isna(cat))
                else (s == cat).to_numpy()
            )
            n_pos = int(y_arr[mask].sum())
            n_total = int(mask.sum())
            kept_records.append((cat, n_pos, n_total - n_pos))

        rare_pos = 0
        rare_neg = 0
        if rare:
            rare_mask = s.isin(rare).to_numpy()
            if any((r is None or pd.isna(r)) for r in rare):
                rare_mask = rare_mask | s.isna().to_numpy()
            rare_pos = int(y_arr[rare_mask].sum())
            rare_neg = int(rare_mask.sum() - rare_pos)

        n_pos_total = sum(p for _, p, _ in kept_records) + rare_pos
        n_neg_total = sum(neg for _, _, neg in kept_records) + rare_neg
        k = len(kept_records) + (1 if rare else 0)
        if k == 0:
            k = 1

        def woe_iv(n_pos: int, n_neg: int) -> tuple[float, float]:
            pct_pos = (n_pos + 0.5) / (n_pos_total + 0.5 * k)
            pct_neg = (n_neg + 0.5) / (n_neg_total + 0.5 * k)
            woe = log(pct_pos / pct_neg)
            iv_contrib = (pct_pos - pct_neg) * woe
            return woe, iv_contrib

        bins: list[WoEBin] = []
        woe_map: dict[Any, float] = {}
        for cat, n_pos, n_neg in kept_records:
            woe, ivc = woe_iv(n_pos, n_neg)
            n_total = n_pos + n_neg
            bins.append(
                WoEBin(
                    label=repr(cat),
                    n_total=n_total,
                    n_pos=n_pos,
                    n_neg=n_neg,
                    pos_rate=(n_pos / n_total) if n_total > 0 else 0.0,
                    woe=woe,
                    iv_contrib=ivc,
                )
            )
            woe_map[cat] = woe

        rare_woe = 0.0
        if rare:
            woe_r, ivc_r = woe_iv(rare_pos, rare_neg)
            rare_woe = woe_r
            n_total_r = rare_pos + rare_neg
            bins.append(
                WoEBin(
                    label="__rare__",
                    n_total=n_total_r,
                    n_pos=rare_pos,
                    n_neg=rare_neg,
                    pos_rate=(rare_pos / n_total_r) if n_total_r > 0 else 0.0,
                    woe=woe_r,
                    iv_contrib=ivc_r,
                )
            )

        iv = float(sum(b.iv_contrib for b in bins))

        self.woe_map_ = woe_map
        self.rare_woe_ = float(rare_woe)
        self.table_ = WoETable(
            column=col,
            kind="categorical",
            bins=bins,
            iv=iv,
            n_pos_total=n_pos_total,
            n_neg_total=n_neg_total,
            nan_woe=0.0,
        )
        self.feature_names_in_ = [col]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("input must be a pandas DataFrame")
        col = X.columns[0]
        s = X[col]
        out = s.map(self.woe_map_)
        out = out.fillna(self.rare_woe_)
        return pd.DataFrame({col: out.astype("float64")}, index=X.index)

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        return list(self.feature_names_in_)
