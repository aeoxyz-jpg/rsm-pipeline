"""PDO scorecard math + WoE table extraction."""

from __future__ import annotations

import math
from typing import Any, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression


def _unwrap_to_lr(model: Any) -> Optional[LogisticRegression]:
    """Drill through CalibratedClassifierCV / FrozenEstimator to find an LR.

    Returns None if no LR is found.
    """
    cur = model
    seen: set[int] = set()
    while id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, LogisticRegression):
            return cur
        # CalibratedClassifierCV: list of calibrated_classifiers_, take first
        if hasattr(cur, "calibrated_classifiers_") and cur.calibrated_classifiers_:
            cur = cur.calibrated_classifiers_[0].estimator
            continue
        # FrozenEstimator: .estimator
        if hasattr(cur, "estimator") and not isinstance(cur, LogisticRegression):
            cur = cur.estimator
            continue
        break
    return None


def _collect_woe_encoders(
    fe_pipeline: Any,
) -> Optional[dict[str, Any]]:
    """Walk a ColumnTransformer; return {output_col: encoder} or None.

    Returns None if any output column was produced by something other than a WoE encoder.
    """
    from rsm_pipeline.feature_engineering.woe.categorical import (
        CategoricalWoEEncoder,
    )
    from rsm_pipeline.feature_engineering.woe.numerical import (
        NumericalWoEEncoder,
    )

    if fe_pipeline is None:
        return None
    if not isinstance(fe_pipeline, ColumnTransformer):
        return None

    out: dict[str, Any] = {}
    for name, transformer, cols in fe_pipeline.transformers_:
        if name == "remainder":
            continue
        if isinstance(transformer, (NumericalWoEEncoder, CategoricalWoEEncoder)):
            # Each WoE encoder we built has one input column → same output col name
            col = list(cols)[0] if hasattr(cols, "__iter__") else cols
            out[str(col)] = transformer
        else:
            return None  # non-WoE component → ineligible
    return out


def _is_pdo_eligible(model: Any, fe_pipeline: Any) -> bool:
    return (
        _unwrap_to_lr(model) is not None
        and _collect_woe_encoders(fe_pipeline) is not None
    )


def _pdo_factor_offset(
    base_score: int, base_odds: float, pdo: int
) -> tuple[float, float]:
    factor = pdo / math.log(2)
    offset = base_score - factor * math.log(base_odds)
    return factor, offset


def _build_pdo_table(
    lr: LogisticRegression,
    woe_encoders: dict[str, Any],
    feature_order: list[str],
    base_score: int,
    base_odds: float,
    pdo: int,
) -> tuple[pd.DataFrame, dict]:
    """Return (table_df, meta) where meta has factor/offset/intercept/n_features."""
    factor, offset = _pdo_factor_offset(base_score, base_odds, pdo)
    intercept = float(lr.intercept_[0])
    weights = lr.coef_[0]
    n_features = len(feature_order)
    if n_features == 0:
        raise ValueError("PDO scorecard requires at least one feature")

    rows: list[dict] = []
    base_per_feature = (offset / n_features) - (intercept * factor / n_features)

    for col, w in zip(feature_order, weights):
        enc = woe_encoders.get(col)
        if enc is None:
            raise KeyError(f"no WoE encoder for feature {col!r}")
        # Each encoder exposes a `table_: WoETable` (cf. feature_engineering.woe.iv).
        for b in enc.table_.bins:
            woe_v = float(b.woe)
            points = round(-float(w) * woe_v * factor + base_per_feature)
            rows.append(
                {
                    "feature": col,
                    "bin_label": b.label,
                    "woe": woe_v,
                    "n_total": int(b.n_total),
                    "n_pos": int(b.n_pos),
                    "n_neg": int(b.n_neg),
                    "pos_rate": float(b.pos_rate),
                    "weight": float(w),
                    "points": int(points),
                }
            )

    df = pd.DataFrame(rows)
    meta = {
        "factor": float(factor),
        "offset": float(offset),
        "intercept": intercept,
        "n_features": n_features,
        "features": list(feature_order),
        "base_score": int(base_score),
        "base_odds": float(base_odds),
        "pdo": int(pdo),
    }
    return df, meta
