"""apply_tuning glue."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from rsm_pipeline.config.schema import RsmConfig
from rsm_pipeline.models.factory import build_model
from rsm_pipeline.tuning.backends.grid import _run_grid
from rsm_pipeline.tuning.backends.optuna_be import _run_optuna
from rsm_pipeline.tuning.backends.random import _run_random

_log = logging.getLogger(__name__)


def apply_tuning(
    cfg: RsmConfig,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    class_weight: dict | None,
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run hyperparameter search; return (best_params, meta)."""
    assert cfg.tuning is not None, "apply_tuning called without cfg.tuning"
    kind = cfg.model.estimator.kind
    if kind in ("voting", "stacking"):
        raise ValueError(
            f"tuning is not supported for ensemble kind={kind!r} in #7b. "
            "Tune individual members in a separate run, or remove cfg.tuning."
        )
    if kind == "dummy":
        raise ValueError("tuning kind='dummy' is not meaningful; remove cfg.tuning.")

    if class_weight is not None:
        from rsm_pipeline.models.apply import _normalize_class_weight

        class_weight = _normalize_class_weight(class_weight)

    base = build_model(
        cfg.model.estimator, class_weight=class_weight, seed=cfg.run.seed
    )

    if cfg.tuning.backend == "grid":
        result = _run_grid(base, cfg.tuning, X, y)
    elif cfg.tuning.backend == "random":
        result = _run_random(base, cfg.tuning, X, y, seed=cfg.run.seed)
    elif cfg.tuning.backend == "optuna":
        result = _run_optuna(base, cfg.tuning, X, y, seed=cfg.run.seed)
    else:
        raise ValueError(f"unsupported tuning backend: {cfg.tuning.backend!r}")

    out_path = run_dir / cfg.tuning.report.history_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    best_params = result["best_params"]
    meta = {
        "backend": result["backend"],
        "n_trials": result["n_trials"],
        "best_score": result["best_score"],
        "best_params": best_params,
        "history_path": cfg.tuning.report.history_path,
    }
    return best_params, meta
