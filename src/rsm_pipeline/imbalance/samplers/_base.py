"""Shared helpers for imbalance samplers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight


def _class_counts(y: pd.Series) -> dict[str, int]:
    """Return a JSON-friendly {str(class_label): int(count)} mapping."""
    return {str(k): int(v) for k, v in y.value_counts().items()}


def _minority_ratio(y: pd.Series) -> float:
    """Return min(count) / total. 1.0 when only one class present (degenerate)."""
    counts = y.value_counts()
    if len(counts) == 0:
        return 0.0
    return float(counts.min()) / float(counts.sum())


def _compute_class_weight(
    weight: Any,
    y: pd.Series,
) -> dict[str, float]:
    """Resolve 'balanced' or a dict to a JSON-friendly weight mapping."""
    classes = np.sort(y.unique())
    if weight == "balanced":
        w = compute_class_weight(class_weight="balanced", classes=classes, y=y.values)
        return {str(c): float(v) for c, v in zip(classes, w)}
    if isinstance(weight, dict):
        known_str = {str(c) for c in classes.tolist()}
        unknown = {k for k in weight.keys() if str(k) not in known_str}
        if unknown:
            raise ValueError(
                f"class_weight has unknown classes {sorted(map(str, unknown))}; "
                f"train classes are {sorted(known_str)}"
            )
        return {str(k): float(v) for k, v in weight.items()}
    raise ValueError(f"unsupported class_weight value: {weight!r}")
