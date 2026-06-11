"""Optuna TPE backend."""

from __future__ import annotations

import logging
from typing import Any

import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

from rsm_pipeline.tuning.schema import TuningConfig
from rsm_pipeline.tuning.space import _apply_to_optuna_trial

_log = logging.getLogger(__name__)


def _run_optuna(
    base_estimator: Any,
    cfg: TuningConfig,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int,
) -> dict:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        params = _apply_to_optuna_trial(cfg.search_space, trial)
        est = clone(base_estimator)
        est.set_params(**params)
        scores = cross_val_score(
            est, X, y, cv=cfg.cv, scoring=cfg.scoring, n_jobs=cfg.n_jobs
        )
        return float(scores.mean())

    study.optimize(
        objective,
        n_trials=cfg.n_trials,
        timeout=cfg.timeout,
        gc_after_trial=True,
    )

    history = []
    for i, t in enumerate(study.trials):
        history.append(
            {
                "trial": i,
                "params": dict(t.params),
                "score": float(t.value) if t.value is not None else None,
                "state": str(t.state).split(".")[-1],
            }
        )

    return {
        "backend": "optuna",
        "n_trials": len(history),
        "best_score": float(study.best_value),
        "best_params": dict(study.best_params),
        "history": history,
    }
