"""Voting / Stacking factories."""

from __future__ import annotations

from typing import Any, Optional

from rsm_pipeline.ensembles.schema import StackingConfig, VotingConfig


def _make_voting(
    cfg: VotingConfig,
    *,
    class_weight: Optional[dict],
    seed: int,
) -> Any:
    from sklearn.ensemble import VotingClassifier

    from rsm_pipeline.models.factory import build_model

    members = [
        (f"m{i}", build_model(m, class_weight=class_weight, seed=seed))
        for i, m in enumerate(cfg.members)
    ]
    return VotingClassifier(
        estimators=members,
        voting=cfg.voting,
        weights=cfg.weights,
        n_jobs=cfg.n_jobs,
    )


def _make_stacking(
    cfg: StackingConfig,
    *,
    class_weight: Optional[dict],
    seed: int,
) -> Any:
    from sklearn.ensemble import StackingClassifier

    from rsm_pipeline.models.factory import build_model

    members = [
        (f"m{i}", build_model(m, class_weight=class_weight, seed=seed))
        for i, m in enumerate(cfg.members)
    ]
    final = build_model(cfg.final_estimator, class_weight=class_weight, seed=seed)
    return StackingClassifier(
        estimators=members,
        final_estimator=final,
        cv=cfg.cv,
        passthrough=cfg.passthrough,
        n_jobs=cfg.n_jobs,
    )
