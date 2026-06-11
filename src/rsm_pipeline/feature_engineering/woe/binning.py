"""Numeric binning: quantile and decision-tree-based monotonic."""

from __future__ import annotations

import logging
from math import inf

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


def quantile_bins(series: pd.Series, n_bins: int) -> list[float]:
    """Return cut points: [-inf, q1, q2, ..., +inf]. Length >= 3 enforced."""
    arr = series.dropna().to_numpy()
    if len(arr) == 0:
        raise ValueError("quantile_bins: series has no non-null values")
    qs = np.nanquantile(arr, np.linspace(0, 1, n_bins + 1))
    cuts_unique = sorted({float(q) for q in qs.tolist()})
    inner = cuts_unique[1:-1] if len(cuts_unique) >= 2 else []
    cuts = [-inf, *inner, inf]
    if len(cuts) < 3:
        raise ValueError(f"column too uniform to bin (only {len(cuts)} cuts produced)")
    return cuts


def monotonic_bins(
    series: pd.Series,
    y: pd.Series,
    n_leaves: int,
    min_bin_pct: float,
) -> list[float]:
    """Decision-tree-driven cut points. Falls back to quantile_bins on failure."""
    from sklearn.tree import DecisionTreeClassifier

    arr = series.to_numpy()
    mask = ~np.isnan(arr)
    arr_clean = arr[mask].reshape(-1, 1)
    y_clean = y.to_numpy()[mask]

    if len(arr_clean) == 0:
        raise ValueError("monotonic_bins: no non-null values")

    min_samples_leaf = max(1, int(len(arr_clean) * min_bin_pct))
    try:
        tree = DecisionTreeClassifier(
            max_leaf_nodes=n_leaves,
            min_samples_leaf=min_samples_leaf,
            random_state=0,
        )
        tree.fit(arr_clean, y_clean)
    except Exception as e:
        _log.warning(
            "monotonic_bins: tree fit failed (%s); falling back to quantile", e
        )
        return quantile_bins(series, n_leaves)

    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    cuts_inner = sorted(
        float(threshold[i]) for i in range(len(threshold)) if feature[i] >= 0
    )
    cuts = [-inf, *cuts_inner, inf]
    if len(cuts) < 3:
        _log.warning(
            "monotonic_bins: only %d cuts; falling back to quantile", len(cuts)
        )
        return quantile_bins(series, n_leaves)
    return cuts
