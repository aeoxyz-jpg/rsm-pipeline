"""Decile lift table — top 10% in 1% bins + remaining 90% in 9 deciles."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _compute_decile_lift(y_true: pd.Series, y_score: np.ndarray) -> pd.DataFrame:
    """19-row decile-lift table sorted by descending score.

    Top 10% split into ten 1%-wide bins (rank 0-1%, 1-2%, ..., 9-10%).
    Remaining 90% split into nine deciles (10-20%, ..., 90-100%).
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_score)

    order = np.argsort(-y_score, kind="stable")
    y_sorted = y_true_arr[order]
    total_pos = int(y_sorted.sum())
    total_neg = n - total_pos
    base_rate = total_pos / n if n > 0 else 0.0

    edges: list[tuple[float, float, str]] = []
    for i in range(10):
        lo = i / 100.0
        hi = (i + 1) / 100.0
        edges.append((lo, hi, f"top_{i + 1}pct"))
    for i in range(1, 10):
        lo = i / 10.0
        hi = (i + 1) / 10.0
        edges.append((lo, hi, f"decile_{i + 1}"))

    cum_pos = 0
    cum_neg = 0
    rows: list[dict] = []
    for idx, (lo, hi, label) in enumerate(edges):
        i_lo = int(np.floor(lo * n))
        i_hi = int(np.floor(hi * n)) if idx < len(edges) - 1 else n
        chunk = y_sorted[i_lo:i_hi]
        n_bin = len(chunk)
        n_pos = int(chunk.sum())
        n_neg = n_bin - n_pos
        cum_pos += n_pos
        cum_neg += n_neg
        cum_pop = i_hi
        precision = (n_pos / n_bin) if n_bin > 0 else 0.0
        recall = (cum_pos / total_pos) if total_pos > 0 else 0.0
        cum_pop_pct = cum_pop / n if n > 0 else 0.0
        cum_pos_pct = recall
        cum_neg_pct = (cum_neg / total_neg) if total_neg > 0 else 0.0
        lift = (precision / base_rate) if base_rate > 0 else 0.0
        cum_lift = (
            (cum_pos / cum_pop) / base_rate if base_rate > 0 and cum_pop > 0 else 0.0
        )
        cum_gain = cum_pos_pct
        rows.append(
            {
                "bin": label,
                "rank_low": float(lo),
                "rank_high": float(hi),
                "n": int(n_bin),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "cum_pop_pct": float(cum_pop_pct),
                "cum_pos_pct": float(cum_pos_pct),
                "cum_neg_pct": float(cum_neg_pct),
                "precision": float(precision),
                "recall": float(recall),
                "lift": float(lift),
                "cum_lift": float(cum_lift),
                "cum_gain": float(cum_gain),
            }
        )
    return pd.DataFrame(rows)
