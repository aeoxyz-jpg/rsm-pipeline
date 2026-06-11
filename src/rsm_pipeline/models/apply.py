"""Glue: fit, evaluate, persist."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from rsm_pipeline.config.schema import RsmConfig
from rsm_pipeline.models.factory import build_model

_log = logging.getLogger(__name__)


def _expand_class_weight_to_sample_weight(
    class_weight: dict, y: pd.Series
) -> np.ndarray:
    """Expand a {'0': w0, '1': w1} dict to a per-sample weight vector."""
    try:
        w0 = float(class_weight[0] if 0 in class_weight else class_weight["0"])
        w1 = float(class_weight[1] if 1 in class_weight else class_weight["1"])
    except KeyError:
        _log.warning("class_weight missing '0'/'1'; using uniform weights")
        return np.ones(len(y), dtype=float)
    return np.where(y.values == 1, w1, w0).astype(float)


def _normalize_class_weight(class_weight: dict) -> dict:
    """Coerce string keys ('0', '1') to int keys (0, 1) for sklearn compatibility."""
    result = {}
    for k, v in class_weight.items():
        try:
            result[int(k)] = float(v)
        except (ValueError, TypeError):
            result[k] = v
    return result


def fit_model(
    cfg: RsmConfig,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    class_weight: Optional[dict] = None,
) -> Any:
    """Build and fit the configured estimator."""
    kind = cfg.model.estimator.kind
    if class_weight is not None:
        class_weight = _normalize_class_weight(class_weight)
    est = build_model(cfg.model.estimator, class_weight=class_weight, seed=cfg.run.seed)
    if class_weight is not None and kind == "mlp":
        _log.warning(
            "mlp does not support class_weight or sample_weight; "
            "fitting without imbalance correction"
        )
        est.fit(X, y)
    elif class_weight is not None and kind == "dummy":
        sw = _expand_class_weight_to_sample_weight(class_weight, y)
        est.fit(X, y, sample_weight=sw)
    else:
        est.fit(X, y)
    return est


def fit_model_with_estimator(
    cfg: RsmConfig,
    estimator_cfg: Any,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    class_weight: Optional[dict] = None,
) -> Any:
    """Variant of fit_model that takes an explicit estimator config."""
    kind = estimator_cfg.kind
    if class_weight is not None:
        class_weight = _normalize_class_weight(class_weight)
    est = build_model(estimator_cfg, class_weight=class_weight, seed=cfg.run.seed)
    if class_weight is not None and kind == "mlp":
        _log.warning(
            "mlp does not support class_weight; fitting without imbalance correction"
        )
        est.fit(X, y)
    elif class_weight is not None and kind == "dummy":
        sw = _expand_class_weight_to_sample_weight(class_weight, y)
        est.fit(X, y, sample_weight=sw)
    else:
        est.fit(X, y)
    return est


def evaluate_model(
    model: Any, X: pd.DataFrame, y: pd.Series, cfg: RsmConfig
) -> dict[str, float]:
    """Compute cfg.evaluation.metrics. Only roc_auc supported in #7a."""
    proba = model.predict_proba(X)
    scores = proba[:, 1] if proba.shape[1] >= 2 else np.full(len(X), 0.5)
    out: dict[str, float] = {}
    for m in cfg.evaluation.metrics:
        if m == "roc_auc":
            out["roc_auc"] = float(roc_auc_score(y, scores))
        else:
            raise ValueError(f"unsupported metric in this spec: {m!r}")
    return out


def persist_model(model: Any, run_dir: Path, cfg: RsmConfig) -> str:
    """Joblib-dump the fitted model. Returns relative path."""
    rel = cfg.model.persist.path
    out = run_dir / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    return rel
