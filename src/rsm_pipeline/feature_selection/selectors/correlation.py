"""Correlation-based column drop (greedy, IV-driven tie-break)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rsm_pipeline.feature_selection.selectors._base import _SelectorBase
from rsm_pipeline.feature_selection.selectors.iv_threshold import _compute_iv


class CorrelationSelector(_SelectorBase):
    def __init__(
        self,
        threshold: float = 0.95,
        method: str = "pearson",
        tie_break: str = "iv",
    ):
        self.threshold = threshold
        self.method = method
        self.tie_break = tie_break

    def _fit_kept(self, X: pd.DataFrame, y: Any) -> list[str]:
        if X.shape[1] < 2:
            return list(X.columns)

        corr = X.corr(method=self.method).abs()
        np.fill_diagonal(corr.values, 0.0)
        self.corr_matrix_ = corr

        if self.tie_break == "iv":
            y_series = (
                pd.Series(y, index=X.index) if not isinstance(y, pd.Series) else y
            )
            scores = {
                c: _compute_iv(X[c], y_series, n_bins=10, smoothing=0.5)
                for c in X.columns
            }
        elif self.tie_break == "first":
            scores = {c: -i for i, c in enumerate(X.columns)}
        else:  # "name"
            # Lexicographic — earlier letters preferred (kept).
            scores = {c: -ord(c[0]) if c else 0 for c in X.columns}
        self.tie_scores_ = scores

        kept = set(X.columns)
        dropped_pairs: list[tuple[str, str, float]] = []
        while True:
            sub = corr.loc[list(kept), list(kept)] if kept else corr.iloc[:0, :0]
            if sub.size == 0:
                break
            max_val = float(sub.values.max())
            if max_val <= self.threshold:
                break
            i, j = np.unravel_index(np.argmax(sub.values), sub.shape)
            a, b = sub.index[i], sub.columns[j]
            if scores[a] != scores[b]:
                loser = b if scores[a] > scores[b] else a
            else:
                loser = max(a, b)
            winner = a if loser != a else b
            kept.discard(loser)
            dropped_pairs.append((loser, winner, max_val))
        self.dropped_pairs_ = dropped_pairs
        return [c for c in X.columns if c in kept]

    def drop_reasons(self) -> dict[str, str]:
        return {
            loser: f"|corr|={v:.4f} > {self.threshold} with {winner}"
            for loser, winner, v in self.dropped_pairs_
        }
