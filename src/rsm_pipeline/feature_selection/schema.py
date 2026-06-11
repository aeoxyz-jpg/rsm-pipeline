"""Pydantic v2 models for feature selection configuration."""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import Field, model_validator

from rsm_pipeline._frozen import _Frozen


class VarianceSelectorConfig(_Frozen):
    kind: Literal["variance"] = "variance"
    threshold: float = 0.0

    @model_validator(mode="after")
    def _validate(self) -> "VarianceSelectorConfig":
        if self.threshold < 0:
            raise ValueError(f"variance threshold must be >= 0 (got {self.threshold})")
        return self


class IVThresholdSelectorConfig(_Frozen):
    kind: Literal["iv_threshold"] = "iv_threshold"
    threshold: float = 0.02
    n_bins: int = 10
    smoothing: float = 0.5

    @model_validator(mode="after")
    def _validate(self) -> "IVThresholdSelectorConfig":
        if self.threshold < 0:
            raise ValueError(f"iv threshold must be >= 0 (got {self.threshold})")
        if self.n_bins < 2:
            raise ValueError(f"n_bins must be >= 2 (got {self.n_bins})")
        if self.smoothing < 0:
            raise ValueError(f"smoothing must be >= 0 (got {self.smoothing})")
        return self


class CorrelationSelectorConfig(_Frozen):
    kind: Literal["correlation"] = "correlation"
    threshold: float = 0.95
    method: Literal["pearson"] = "pearson"
    tie_break: Literal["iv", "first", "name"] = "iv"

    @model_validator(mode="after")
    def _validate(self) -> "CorrelationSelectorConfig":
        if not (0 < self.threshold <= 1):
            raise ValueError(
                f"correlation threshold must satisfy 0 < x <= 1 (got {self.threshold})"
            )
        return self


class ImportanceSelectorConfig(_Frozen):
    kind: Literal["importance"] = "importance"
    top_k: Optional[int] = None
    top_k_pct: Optional[float] = None
    n_estimators: int = 200
    max_depth: Optional[int] = 8
    random_state: Optional[int] = None

    @model_validator(mode="after")
    def _validate(self) -> "ImportanceSelectorConfig":
        if (self.top_k is None) == (self.top_k_pct is None):
            raise ValueError("exactly one of top_k or top_k_pct must be set")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError(f"top_k must be >= 1 (got {self.top_k})")
        if self.top_k_pct is not None and not (0 < self.top_k_pct <= 1):
            raise ValueError(
                f"top_k_pct must satisfy 0 < x <= 1 (got {self.top_k_pct})"
            )
        if self.n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1 (got {self.n_estimators})")
        if self.max_depth is not None and self.max_depth < 1:
            raise ValueError(f"max_depth must be >= 1 (got {self.max_depth})")
        return self


class RFESelectorConfig(_Frozen):
    kind: Literal["rfe"] = "rfe"
    n_features_to_select: int
    step: int = 1

    @model_validator(mode="after")
    def _validate(self) -> "RFESelectorConfig":
        if self.n_features_to_select < 1:
            raise ValueError(
                f"n_features_to_select must be >= 1 (got {self.n_features_to_select})"
            )
        if self.step < 1:
            raise ValueError(f"step must be >= 1 (got {self.step})")
        return self


SelectorConfig = Annotated[
    Union[
        VarianceSelectorConfig,
        IVThresholdSelectorConfig,
        CorrelationSelectorConfig,
        ImportanceSelectorConfig,
        RFESelectorConfig,
    ],
    Field(discriminator="kind"),
]


class FeatureSelectionReportConfig(_Frozen):
    csv_path: str = "reports/feature_selection.csv"


class FeatureSelectionConfig(_Frozen):
    selectors: list[SelectorConfig] = Field(default_factory=list)
    report: FeatureSelectionReportConfig = Field(
        default_factory=FeatureSelectionReportConfig
    )
