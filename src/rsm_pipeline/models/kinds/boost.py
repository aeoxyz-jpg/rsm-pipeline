"""XGBoost / LightGBM / CatBoost factories."""

from __future__ import annotations

import logging
from typing import Optional

from rsm_pipeline.models.schema import (
    CatBoostConfig,
    LightGBMConfig,
    XGBoostConfig,
)

_log = logging.getLogger(__name__)


def _scale_pos_weight_from(class_weight: Optional[dict]) -> float:
    """Convert canonical {'0': w0, '1': w1} (or {0,1}) to scale_pos_weight."""
    if class_weight is None:
        return 1.0
    try:
        w0 = float(class_weight[0] if 0 in class_weight else class_weight["0"])
        w1 = float(class_weight[1] if 1 in class_weight else class_weight["1"])
        return w1 / w0 if w0 > 0 else 1.0
    except KeyError:
        _log.warning(
            "class_weight missing keys '0'/'1'; defaulting scale_pos_weight=1.0"
        )
        return 1.0


def _normalize_class_weight_keys(class_weight: Optional[dict]) -> Optional[dict]:
    """Cast keys to int (LightGBM expects int class labels)."""
    if class_weight is None:
        return None
    try:
        return {int(k): float(v) for k, v in class_weight.items()}
    except (ValueError, TypeError):
        _log.warning("class_weight keys not int-castable; passing through unchanged")
        return class_weight


def _make_xgboost(
    cfg: XGBoostConfig,
    class_weight: Optional[dict],
    *,
    seed: int,
):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        n_jobs=cfg.n_jobs,
        tree_method=cfg.tree_method,
        scale_pos_weight=_scale_pos_weight_from(class_weight),
        random_state=seed,
        eval_metric="logloss",
    )


def _make_lightgbm(
    cfg: LightGBMConfig,
    class_weight: Optional[dict],
    *,
    seed: int,
):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=cfg.n_estimators,
        num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        n_jobs=cfg.n_jobs,
        class_weight=_normalize_class_weight_keys(class_weight),
        random_state=seed,
        verbose=-1,
    )


def _make_catboost(
    cfg: CatBoostConfig,
    class_weight: Optional[dict],
    *,
    seed: int,
):
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise RuntimeError(
            "catboost is an optional extra. Install with "
            "`pip install '.[catboost]'` or choose a different model kind."
        ) from exc

    cw_list = None
    if class_weight is not None:
        try:
            w0 = float(class_weight[0] if 0 in class_weight else class_weight["0"])
            w1 = float(class_weight[1] if 1 in class_weight else class_weight["1"])
            cw_list = [w0, w1]
        except KeyError:
            _log.warning("catboost class_weights skipped (missing '0'/'1')")

    return CatBoostClassifier(
        iterations=cfg.iterations,
        depth=cfg.depth,
        learning_rate=cfg.learning_rate,
        verbose=cfg.verbose,
        class_weights=cw_list,
        random_seed=seed,
    )
