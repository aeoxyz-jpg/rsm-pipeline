"""Tuning configs (search space + backend selection)."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import Field, model_validator

from rsm_pipeline._frozen import _Frozen


class _DistBase(_Frozen):
    pass


class FloatDist(_DistBase):
    type: Literal["float"] = "float"
    low: float
    high: float
    log: bool = False
    step: Optional[float] = None


class IntDist(_DistBase):
    type: Literal["int"] = "int"
    low: int
    high: int
    step: int = 1
    log: bool = False


class CategoricalDist(_DistBase):
    type: Literal["categorical"] = "categorical"
    choices: list[Any]


Distribution = Annotated[
    Union[FloatDist, IntDist, CategoricalDist],
    Field(discriminator="type"),
]


SearchSpaceValue = Union[Distribution, list[Any]]


class TuningReportConfig(_Frozen):
    history_path: str = "reports/tuning_history.json"


class TuningConfig(_Frozen):
    backend: Literal["grid", "random", "optuna"]
    search_space: dict[str, SearchSpaceValue]
    cv: int = 5
    scoring: Literal["roc_auc"] = "roc_auc"
    n_jobs: int = -1
    n_trials: int = 50
    timeout: Optional[int] = None
    refit: bool = True
    report: TuningReportConfig = Field(default_factory=TuningReportConfig)

    @model_validator(mode="after")
    def _backend_constraints(self) -> "TuningConfig":
        if self.backend == "grid":
            for k, v in self.search_space.items():
                if not isinstance(v, list):
                    raise ValueError(
                        f"grid backend requires list-valued search_space[{k!r}] "
                        f"(got {type(v).__name__})"
                    )
        return self
