"""Pydantic v2 configs for imbalance handling. Frozen, extra='forbid'."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import Field, model_validator

from rsm_pipeline._frozen import _Frozen


class _SamplerBase(_Frozen):
    pass


class NoneSamplerConfig(_SamplerBase):
    kind: Literal["none"] = "none"


class ClassWeightConfig(_SamplerBase):
    kind: Literal["class_weight"] = "class_weight"
    weight: Union[Literal["balanced"], dict[Any, float]] = "balanced"


class RandomOverConfig(_SamplerBase):
    kind: Literal["random_oversample"] = "random_oversample"
    sampling_strategy: Union[float, str, dict[Any, int]] = "auto"


class RandomUnderConfig(_SamplerBase):
    kind: Literal["random_undersample"] = "random_undersample"
    sampling_strategy: Union[float, str, dict[Any, int]] = "auto"


class SmoteConfig(_SamplerBase):
    kind: Literal["smote"] = "smote"
    sampling_strategy: Union[float, str, dict[Any, int]] = "auto"
    k_neighbors: int = 5
    categorical_features: Optional[list[str]] = None


class AdasynConfig(_SamplerBase):
    kind: Literal["adasyn"] = "adasyn"
    sampling_strategy: Union[float, str, dict[Any, int]] = "auto"
    n_neighbors: int = 5


class SmoteennConfig(_SamplerBase):
    kind: Literal["smoteenn"] = "smoteenn"
    sampling_strategy: Union[float, str, dict[Any, int]] = "auto"


class SmotetomekConfig(_SamplerBase):
    kind: Literal["smotetomek"] = "smotetomek"
    sampling_strategy: Union[float, str, dict[Any, int]] = "auto"


class AutoConfig(_SamplerBase):
    kind: Literal["auto"] = "auto"
    high_imbalance_threshold: float = 0.1
    moderate_imbalance_threshold: float = 0.33

    @model_validator(mode="after")
    def _ordered(self) -> "AutoConfig":
        if not (
            0.0
            < self.high_imbalance_threshold
            < self.moderate_imbalance_threshold
            < 0.5
        ):
            raise ValueError(
                "auto thresholds must satisfy "
                "0 < high_imbalance_threshold < moderate_imbalance_threshold < 0.5"
            )
        return self


SamplerConfig = Annotated[
    Union[
        NoneSamplerConfig,
        ClassWeightConfig,
        RandomOverConfig,
        RandomUnderConfig,
        SmoteConfig,
        AdasynConfig,
        SmoteennConfig,
        SmotetomekConfig,
        AutoConfig,
    ],
    Field(discriminator="kind"),
]


class ImbalanceReportConfig(_Frozen):
    json_path: str = "imbalance_report.json"


class ImbalanceConfig(_Frozen):
    sampler: SamplerConfig = Field(default_factory=NoneSamplerConfig)
    report: ImbalanceReportConfig = Field(default_factory=ImbalanceReportConfig)
