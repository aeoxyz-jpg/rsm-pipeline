"""Factory dispatch from ModelKindConfig to a fittable estimator."""

from __future__ import annotations

from typing import Any, Optional

from rsm_pipeline.models.kinds._dummy import _make_dummy
from rsm_pipeline.models.kinds.boost import (
    _make_catboost,
    _make_lightgbm,
    _make_xgboost,
)
from rsm_pipeline.models.kinds.linear import _make_logreg
from rsm_pipeline.models.kinds.mlp import _make_mlp
from rsm_pipeline.models.kinds.tree import _make_extra_trees, _make_random_forest
from rsm_pipeline.ensembles.schema import StackingConfig, VotingConfig
from rsm_pipeline.models.schema import (
    CatBoostConfig,
    DummyModelConfig,
    ExtraTreesConfig,
    LightGBMConfig,
    LogRegConfig,
    MLPConfig,
    ModelKindConfig,
    RandomForestConfig,
    XGBoostConfig,
)


def build_model(
    cfg: ModelKindConfig,
    *,
    class_weight: Optional[dict] = None,
    seed: int,
) -> Any:
    """Dispatch by ``cfg.kind`` to the appropriate factory."""
    if isinstance(cfg, DummyModelConfig):
        return _make_dummy(cfg, seed=seed)
    if isinstance(cfg, LogRegConfig):
        return _make_logreg(cfg, class_weight, seed=seed)
    if isinstance(cfg, RandomForestConfig):
        return _make_random_forest(cfg, class_weight, seed=seed)
    if isinstance(cfg, ExtraTreesConfig):
        return _make_extra_trees(cfg, class_weight, seed=seed)
    if isinstance(cfg, XGBoostConfig):
        return _make_xgboost(cfg, class_weight, seed=seed)
    if isinstance(cfg, LightGBMConfig):
        return _make_lightgbm(cfg, class_weight, seed=seed)
    if isinstance(cfg, CatBoostConfig):
        return _make_catboost(cfg, class_weight, seed=seed)
    if isinstance(cfg, MLPConfig):
        return _make_mlp(cfg, seed=seed)
    if isinstance(cfg, VotingConfig):
        from rsm_pipeline.ensembles.factory import _make_voting

        return _make_voting(cfg, class_weight=class_weight, seed=seed)
    if isinstance(cfg, StackingConfig):
        from rsm_pipeline.ensembles.factory import _make_stacking

        return _make_stacking(cfg, class_weight=class_weight, seed=seed)
    raise TypeError(f"unsupported estimator config: {type(cfg).__name__}")
