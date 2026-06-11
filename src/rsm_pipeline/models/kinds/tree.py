"""Random Forest / Extra Trees factories."""

from __future__ import annotations

from typing import Optional

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from rsm_pipeline.models.schema import ExtraTreesConfig, RandomForestConfig


def _make_random_forest(
    cfg: RandomForestConfig,
    class_weight: Optional[dict],
    *,
    seed: int,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        n_jobs=cfg.n_jobs,
        class_weight=class_weight,
        random_state=seed,
    )


def _make_extra_trees(
    cfg: ExtraTreesConfig,
    class_weight: Optional[dict],
    *,
    seed: int,
) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        n_jobs=cfg.n_jobs,
        class_weight=class_weight,
        random_state=seed,
    )
