"""CSI per feature — numeric quantile binning + categorical level matching."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rsm_pipeline.monitoring.psi import _compute_psi, _tier


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def _csi_categorical(
    ref: pd.Series, cur: pd.Series
) -> tuple[float, list[dict], list[Any]]:
    ref_vals = ref.dropna().astype(str)
    cur_vals = cur.dropna().astype(str)
    cats = sorted(ref_vals.unique().tolist())
    new_levels = sorted(set(cur_vals.unique()) - set(cats))
    # Map unseen → __unseen__ for current
    cur_mapped = cur_vals.where(cur_vals.isin(cats), other="__unseen__")
    bins = cats + (["__unseen__"] if new_levels else [])
    p_ref = (
        np.array([(ref_vals == c).mean() if c in cats else 0.0 for c in bins]) + 1e-6
    )
    p_cur = np.array([(cur_mapped == c).mean() for c in bins]) + 1e-6
    terms = (p_cur - p_ref) * np.log(p_cur / p_ref)
    csi = float(terms.sum())
    breakdown = [
        {
            "level": str(b),
            "p_ref": float(p_ref[i] - 1e-6),
            "p_curr": float(p_cur[i] - 1e-6),
            "contribution": float(terms[i]),
        }
        for i, b in enumerate(bins)
    ]
    return csi, breakdown, new_levels


def _compute_csi_per_feature(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str],
    *,
    n_bins: int,
    minor: float,
    major: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for col in columns:
        ref_s = reference[col]
        cur_s = current[col]
        if _is_numeric(ref_s):
            ref_arr = ref_s.dropna().to_numpy()
            cur_arr = cur_s.dropna().to_numpy()
            if len(ref_arr) == 0 or len(cur_arr) == 0:
                rows.append(
                    {
                        "feature": col,
                        "dtype": "numeric",
                        "csi": None,
                        "tier": "skipped",
                        "reason": "empty after dropna",
                    }
                )
                continue
            csi, _bd = _compute_psi(ref_arr, cur_arr, n_bins=n_bins)
            rows.append(
                {
                    "feature": col,
                    "dtype": "numeric",
                    "csi": csi,
                    "tier": _tier(csi, minor, major),
                    "new_levels": [],
                }
            )
        else:
            csi, _bd, new_levels = _csi_categorical(ref_s, cur_s)
            rows.append(
                {
                    "feature": col,
                    "dtype": "categorical",
                    "csi": csi,
                    "tier": _tier(csi, minor, major),
                    "new_levels": new_levels,
                }
            )
    return rows
