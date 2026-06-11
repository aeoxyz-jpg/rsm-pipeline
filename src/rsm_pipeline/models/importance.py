"""Extract feature importances uniformly across estimator kinds."""

from __future__ import annotations

from typing import Any

import numpy as np


def _get_feature_importances(model: Any, feature_names: list[str]) -> dict[str, float]:
    """Return {feature_name: importance} or {} if model exposes neither."""
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
        if imp.size == len(feature_names):
            return {n: float(v) for n, v in zip(feature_names, imp.tolist())}
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float).ravel()
        if coef.size == len(feature_names):
            return {n: float(v) for n, v in zip(feature_names, np.abs(coef).tolist())}
    return {}
