"""Pydantic v2 models for preprocessing configuration.

All models are frozen and reject extra fields. Conditional rules
(e.g. ``missing == "custom"`` requires ``custom_fill``) are enforced
on the typed defaults; ``ColumnOverride`` validates the same rules
only when its corresponding fields are non-null.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import Field, model_validator

from rsm_pipeline._frozen import _Frozen

_NUMERIC_MISSING = Literal["mean", "median", "knn", "custom", "none"]
_CATEGORICAL_MISSING = Literal["mode", "custom", "none"]
_ANY_MISSING = Literal["mean", "median", "mode", "knn", "custom", "none"]
_OUTLIER = Literal["iqr", "percentile", "zscore", "none"]


def _validate_numeric_block(
    *,
    missing: str,
    custom_fill: Any,
    knn_k: int,
    outlier: str,
    iqr_factor: float,
    percentile_low: float,
    percentile_high: float,
    zscore_threshold: float,
) -> None:
    if missing == "custom" and custom_fill is None:
        raise ValueError("missing=='custom' requires custom_fill to be non-null")
    if missing == "knn" and knn_k < 1:
        raise ValueError(f"missing=='knn' requires knn_k >= 1 (got {knn_k})")
    if outlier == "iqr" and iqr_factor <= 0:
        raise ValueError(f"outlier=='iqr' requires iqr_factor > 0 (got {iqr_factor})")
    if outlier == "percentile":
        if not (0 <= percentile_low < 0.5):
            raise ValueError(
                f"outlier=='percentile' requires 0 <= percentile_low < 0.5 "
                f"(got {percentile_low})"
            )
        if not (0.5 < percentile_high <= 1):
            raise ValueError(
                f"outlier=='percentile' requires 0.5 < percentile_high <= 1 "
                f"(got {percentile_high})"
            )
        if percentile_low >= percentile_high:
            raise ValueError(
                f"percentile_low ({percentile_low}) must be < percentile_high "
                f"({percentile_high})"
            )
    if outlier == "zscore" and zscore_threshold <= 0:
        raise ValueError(
            f"outlier=='zscore' requires zscore_threshold > 0 "
            f"(got {zscore_threshold})"
        )


class NumericDefaults(_Frozen):
    missing: _NUMERIC_MISSING = "median"
    add_indicator: bool = False
    custom_fill: Any | None = None
    knn_k: int = 5
    outlier: _OUTLIER = "iqr"
    iqr_factor: float = 1.5
    percentile_low: float = 0.01
    percentile_high: float = 0.99
    zscore_threshold: float = 3.0

    @model_validator(mode="after")
    def _validate(self) -> "NumericDefaults":
        _validate_numeric_block(
            missing=self.missing,
            custom_fill=self.custom_fill,
            knn_k=self.knn_k,
            outlier=self.outlier,
            iqr_factor=self.iqr_factor,
            percentile_low=self.percentile_low,
            percentile_high=self.percentile_high,
            zscore_threshold=self.zscore_threshold,
        )
        return self


class CategoricalDefaults(_Frozen):
    missing: _CATEGORICAL_MISSING = "mode"
    add_indicator: bool = False
    custom_fill: Any | None = None

    @model_validator(mode="after")
    def _validate(self) -> "CategoricalDefaults":
        if self.missing == "custom" and self.custom_fill is None:
            raise ValueError("missing=='custom' requires custom_fill to be non-null")
        return self


class ColumnOverride(_Frozen):
    treat_as: Optional[Literal["numeric", "categorical"]] = None
    missing: Optional[_ANY_MISSING] = None
    add_indicator: Optional[bool] = None
    custom_fill: Any | None = None
    knn_k: Optional[int] = None
    outlier: Optional[_OUTLIER] = None
    iqr_factor: Optional[float] = None
    percentile_low: Optional[float] = None
    percentile_high: Optional[float] = None
    zscore_threshold: Optional[float] = None

    @model_validator(mode="after")
    def _validate(self) -> "ColumnOverride":
        if self.missing == "custom" and self.custom_fill is None:
            raise ValueError("missing=='custom' requires custom_fill to be non-null")
        if self.missing == "knn" and self.knn_k is not None and self.knn_k < 1:
            raise ValueError(f"knn_k must be >= 1 (got {self.knn_k})")
        if (
            self.outlier == "iqr"
            and self.iqr_factor is not None
            and self.iqr_factor <= 0
        ):
            raise ValueError(f"iqr_factor must be > 0 (got {self.iqr_factor})")
        if self.outlier == "percentile":
            low = self.percentile_low
            high = self.percentile_high
            if low is not None and not (0 <= low < 0.5):
                raise ValueError(
                    f"percentile_low must satisfy 0 <= low < 0.5 (got {low})"
                )
            if high is not None and not (0.5 < high <= 1):
                raise ValueError(
                    f"percentile_high must satisfy 0.5 < high <= 1 (got {high})"
                )
            if low is not None and high is not None and low >= high:
                raise ValueError(
                    f"percentile_low ({low}) must be < percentile_high ({high})"
                )
        if (
            self.outlier == "zscore"
            and self.zscore_threshold is not None
            and self.zscore_threshold <= 0
        ):
            raise ValueError(
                f"zscore_threshold must be > 0 (got {self.zscore_threshold})"
            )
        return self


class PreprocessingConfig(_Frozen):
    numeric: NumericDefaults = Field(default_factory=NumericDefaults)
    categorical: CategoricalDefaults = Field(default_factory=CategoricalDefaults)
    overrides: dict[str, ColumnOverride] = Field(default_factory=dict)
