"""Grid search backend."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import GridSearchCV

from rsm_pipeline.tuning.schema import TuningConfig
from rsm_pipeline.tuning.space import _to_sklearn_grid


def _run_grid(
    base_estimator: Any,
    cfg: TuningConfig,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict:
    grid = _to_sklearn_grid(cfg.search_space)
    cv = GridSearchCV(
        estimator=base_estimator,
        param_grid=grid,
        cv=cfg.cv,
        scoring=cfg.scoring,
        n_jobs=cfg.n_jobs,
        refit=cfg.refit,
    )
    cv.fit(X, y)
    history = [
        {"trial": i, "params": p, "score": float(s)}
        for i, (p, s) in enumerate(
            zip(cv.cv_results_["params"], cv.cv_results_["mean_test_score"])
        )
    ]
    return {
        "backend": "grid",
        "n_trials": len(history),
        "best_score": float(cv.best_score_),
        "best_params": dict(cv.best_params_),
        "history": history,
    }
