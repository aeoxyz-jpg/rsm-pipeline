"""Pydantic v2 configs for the model factory."""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import Field

from rsm_pipeline._frozen import _Frozen


class _ModelKindBase(_Frozen):
    pass


class DummyModelConfig(_ModelKindBase):
    kind: Literal["dummy"] = "dummy"
    strategy: Literal["stratified", "most_frequent", "prior", "uniform", "constant"] = (
        "stratified"
    )
    constant: Optional[int] = None


class LogRegConfig(_ModelKindBase):
    kind: Literal["logreg"] = "logreg"
    C: float = 1.0
    max_iter: int = 1000
    solver: Literal["lbfgs", "liblinear", "saga"] = "lbfgs"


class RandomForestConfig(_ModelKindBase):
    kind: Literal["random_forest"] = "random_forest"
    n_estimators: int = 200
    max_depth: Optional[int] = None
    min_samples_leaf: int = 1
    n_jobs: int = -1


class ExtraTreesConfig(_ModelKindBase):
    kind: Literal["extra_trees"] = "extra_trees"
    n_estimators: int = 200
    max_depth: Optional[int] = None
    min_samples_leaf: int = 1
    n_jobs: int = -1


class XGBoostConfig(_ModelKindBase):
    kind: Literal["xgboost"] = "xgboost"
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    n_jobs: int = -1
    tree_method: Literal["hist", "auto", "exact"] = "hist"


class LightGBMConfig(_ModelKindBase):
    kind: Literal["lightgbm"] = "lightgbm"
    n_estimators: int = 300
    num_leaves: int = 63
    max_depth: int = -1
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    n_jobs: int = -1


class CatBoostConfig(_ModelKindBase):
    kind: Literal["catboost"] = "catboost"
    iterations: int = 500
    depth: int = 6
    learning_rate: float = 0.05
    verbose: int = 0


class MLPConfig(_ModelKindBase):
    kind: Literal["mlp"] = "mlp"
    hidden_layer_sizes: tuple[int, ...] = (64, 32)
    activation: Literal["relu", "tanh", "logistic"] = "relu"
    max_iter: int = 200
    early_stopping: bool = True


NonEnsembleKindConfig = Annotated[
    Union[
        DummyModelConfig,
        LogRegConfig,
        RandomForestConfig,
        ExtraTreesConfig,
        XGBoostConfig,
        LightGBMConfig,
        CatBoostConfig,
        MLPConfig,
    ],
    Field(discriminator="kind"),
]


# Full union with ensembles is resolved via forward ref + model_rebuild()
# (ensembles.schema imports NonEnsembleKindConfig from this module).
ModelKindConfig = Annotated[
    Union[
        DummyModelConfig,
        LogRegConfig,
        RandomForestConfig,
        ExtraTreesConfig,
        XGBoostConfig,
        LightGBMConfig,
        CatBoostConfig,
        MLPConfig,
        "VotingConfig",
        "StackingConfig",
    ],
    Field(discriminator="kind"),
]


class ModelPersistConfig(_Frozen):
    path: str = "artifacts/model.joblib"


class ModelConfig(_Frozen):
    estimator: ModelKindConfig = Field(default_factory=DummyModelConfig)
    persist: ModelPersistConfig = Field(default_factory=ModelPersistConfig)


def _rebuild_with_ensembles() -> None:
    """Resolve VotingConfig/StackingConfig forward refs in ModelKindConfig."""
    from rsm_pipeline.ensembles.schema import (  # noqa: F401
        StackingConfig,
        VotingConfig,
    )

    ModelConfig.model_rebuild()


_rebuild_with_ensembles()
