"""Unwrap CalibratedClassifierCV / FrozenEstimator to leaf estimator."""

from __future__ import annotations

from typing import Any


def _unwrap_to_base_model(model: Any) -> Any:
    cur = model
    seen: set[int] = set()
    while id(cur) not in seen:
        seen.add(id(cur))
        if hasattr(cur, "calibrated_classifiers_") and cur.calibrated_classifiers_:
            cur = cur.calibrated_classifiers_[0].estimator
            continue
        if type(cur).__name__ == "FrozenEstimator" and hasattr(cur, "estimator"):
            cur = cur.estimator
            continue
        break
    return cur
