"""Pydantic v2 evaluation config — full metric panel + plot toggles."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from rsm_pipeline._frozen import _Frozen


class EvaluationPlotsConfig(_Frozen):
    enabled: bool = True
    formats: list[Literal["png"]] = Field(default_factory=lambda: ["png"])


class EvaluationConfig(_Frozen):
    metrics: list[
        Literal[
            "roc_auc",
            "ks",
            "gini",
            "pr_auc",
            "f1",
            "precision",
            "recall",
            "brier",
            "log_loss",
        ]
    ] = Field(
        default_factory=lambda: [
            "roc_auc",
            "ks",
            "gini",
            "pr_auc",
            "brier",
            "log_loss",
        ]
    )
    thresholds: list[float] = Field(default_factory=lambda: [0.5])
    plots: EvaluationPlotsConfig = Field(default_factory=EvaluationPlotsConfig)

    @field_validator("metrics", mode="after")
    @classmethod
    def _non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("evaluation.metrics must contain at least one metric")
        return v

    @field_validator("thresholds", mode="after")
    @classmethod
    def _thresholds_in_range(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("evaluation.thresholds must contain at least one value")
        for t in v:
            if not (0.0 < t < 1.0):
                raise ValueError(f"threshold must be in (0, 1); got {t!r}")
        return v
