"""Logistic Regression factory."""

from __future__ import annotations

from typing import Optional

from sklearn.linear_model import LogisticRegression

from rsm_pipeline.models.schema import LogRegConfig


def _make_logreg(
    cfg: LogRegConfig,
    class_weight: Optional[dict],
    *,
    seed: int,
) -> LogisticRegression:
    return LogisticRegression(
        C=cfg.C,
        max_iter=cfg.max_iter,
        solver=cfg.solver,
        class_weight=class_weight,
        random_state=seed,
    )
