"""Performance drift — recompute #8 metrics on current; diff vs reference JSON."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from rsm_pipeline.evaluation.decile import _compute_decile_lift
from rsm_pipeline.evaluation.metrics import _compute_scalar_metrics


_KEYS = ("roc_auc", "ks", "gini", "pr_auc", "brier", "log_loss")


def _performance_drift(
    bundle: Any,
    current: pd.DataFrame,
    target: str,
    reference_metrics: Optional[dict],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return (summary, decile_df). decile_df is None if no labels usable."""
    if target not in current.columns:
        raise ValueError(
            f"--has-labels was set but target column {target!r} is missing"
        )
    df = current.dropna(subset=[target]).copy()
    if len(df) == 0:
        raise ValueError(f"after dropping rows with NaN target, no rows remain")
    y_true = df[target].astype(int)
    proba = bundle.predict_proba(df.drop(columns=[target], errors="ignore"))
    y_score = proba[:, 1]
    metrics = _compute_scalar_metrics(y_true, y_score, thresholds=[0.5])
    decile = _compute_decile_lift(y_true, y_score)
    summary: dict[str, Any] = {
        "current_metrics": {k: metrics[k] for k in _KEYS},
        "n_current_with_labels": int(len(df)),
    }
    if reference_metrics:
        deltas: dict[str, float] = {}
        for k in _KEYS:
            ref_v = reference_metrics.get(k)
            if ref_v is None:
                continue
            deltas[k] = float(metrics[k] - ref_v)
        summary["reference_metrics"] = {k: reference_metrics.get(k) for k in _KEYS}
        summary["deltas"] = deltas
        # Tier on delta_roc_auc
        d = deltas.get("roc_auc")
        if d is None:
            summary["tier"] = "no_reference_roc_auc"
        elif d <= -0.05:
            summary["tier"] = "major_drop"
        elif d <= -0.01:
            summary["tier"] = "minor_drop"
        else:
            summary["tier"] = "stable"
    else:
        summary["tier"] = "no_reference"
    return summary, decile
