"""Explain config — SHAP + PDP + importance."""

from __future__ import annotations

from pydantic import Field, model_validator

from rsm_pipeline._frozen import _Frozen


class ExplainShapConfig(_Frozen):
    enabled: bool = True
    n_background: int = 100
    n_target: int = 500


class ExplainPdpConfig(_Frozen):
    enabled: bool = True
    n_samples: int = 500
    grid_resolution: int = 50


class ExplainConfig(_Frozen):
    top_k_features: int = 10
    n_local_samples: int = 5
    output_subdir: str = "reports/explain"
    shap: ExplainShapConfig = Field(default_factory=ExplainShapConfig)
    pdp: ExplainPdpConfig = Field(default_factory=ExplainPdpConfig)

    @model_validator(mode="after")
    def _ranges(self) -> "ExplainConfig":
        if self.top_k_features <= 0:
            raise ValueError("explain.top_k_features must be positive")
        if self.n_local_samples < 0:
            raise ValueError("explain.n_local_samples must be non-negative")
        return self
