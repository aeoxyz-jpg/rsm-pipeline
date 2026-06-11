"""Randomized search backend."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from rsm_pipeline.tuning.schema import TuningConfig
from rsm_pipeline.tuning.space import _to_sklearn_distributions


def _run_random(
    base_estimator: Any,
    cfg: TuningConfig,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int,
) -> dict:
    dists = _to_sklearn_distributions(cfg.search_space)
    cv = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=dists,
        n_iter=cfg.n_trials,
        cv=cfg.cv,
        scoring=cfg.scoring,
        n_jobs=cfg.n_jobs,
        refit=cfg.refit,
        random_state=seed,
    )
    cv.fit(X, y)
    history = [
        {"trial": i, "params": p, "score": float(s)}
        for i, (p, s) in enumerate(
            zip(cv.cv_results_["params"], cv.cv_results_["mean_test_score"])
        )
    ]
    return {
        "backend": "random",
        "n_trials": len(history),
        "best_score": float(cv.best_score_),
        "best_params": dict(cv.best_params_),
        "history": history,
    }
