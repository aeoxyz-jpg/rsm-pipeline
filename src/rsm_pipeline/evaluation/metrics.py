"""Scalar metric computation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _clip_proba(p: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    return np.clip(p, eps, 1.0 - eps)


def _compute_scalar_metrics(
    y_true: pd.Series, y_score: np.ndarray, thresholds: list[float]
) -> dict[str, Any]:
    """Compute the full scalar metric panel for a single fold."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    auc = float(roc_auc_score(y_true_arr, y_score))
    fpr, tpr, _ = roc_curve(y_true_arr, y_score)
    out: dict[str, Any] = {
        "roc_auc": auc,
        "pr_auc": float(average_precision_score(y_true_arr, y_score)),
        "brier": float(brier_score_loss(y_true_arr, y_score)),
        "log_loss": float(log_loss(y_true_arr, _clip_proba(y_score))),
        "ks": float(np.max(np.abs(tpr - fpr))),
        "gini": 2 * auc - 1,
        "per_threshold": {},
    }
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        out["per_threshold"][f"{t:.2f}"] = {
            "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        }
    return out
