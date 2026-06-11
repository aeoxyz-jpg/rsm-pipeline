"""Pydantic v2 models for feature engineering configuration."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import Field, model_validator

from rsm_pipeline._frozen import _Frozen

_NUMERIC_METHOD = Literal["standard", "minmax", "robust", "woe", "none"]
_CATEGORICAL_METHOD = Literal["woe", "one_hot", "ordinal", "target", "none"]
_ANY_METHOD = Literal[
    "standard", "minmax", "robust", "woe", "one_hot", "ordinal", "target", "none"
]
_BINNING = Literal["quantile", "monotonic"]


class NumericFEDefaults(_Frozen):
    method: _NUMERIC_METHOD = "standard"
    woe_binning: _BINNING = "quantile"
    woe_n_bins: int = 10
    woe_min_bin_pct: float = 0.05
    woe_min_iv: float = 0.0
    minmax_range: tuple[float, float] = (0.0, 1.0)
    robust_quantile_range: tuple[float, float] = (0.25, 0.75)

    @model_validator(mode="after")
    def _validate(self) -> "NumericFEDefaults":
        if self.method == "woe":
            if self.woe_n_bins < 2:
                raise ValueError(f"woe_n_bins must be >= 2 (got {self.woe_n_bins})")
            if not (0 < self.woe_min_bin_pct < 0.5):
                raise ValueError(
                    f"woe_min_bin_pct must satisfy 0 < x < 0.5 (got {self.woe_min_bin_pct})"
                )
            if self.woe_min_iv < 0:
                raise ValueError(f"woe_min_iv must be >= 0 (got {self.woe_min_iv})")
        if self.method == "minmax":
            lo, hi = self.minmax_range
            if not lo < hi:
                raise ValueError(
                    f"minmax_range must satisfy low < high (got {(lo, hi)})"
                )
        if self.method == "robust":
            qlo, qhi = self.robust_quantile_range
            if not (0 <= qlo < qhi <= 1):
                raise ValueError(
                    f"robust_quantile_range must satisfy 0 <= low < high <= 1 (got {(qlo, qhi)})"
                )
        return self


class CategoricalFEDefaults(_Frozen):
    method: _CATEGORICAL_METHOD = "woe"
    min_bin_pct: float = 0.01
    smoothing: float = 0.0
    one_hot_drop: Optional[Literal["first", "if_binary"]] = None
    ordinal_handle_unknown: Literal["use_encoded_value", "error"] = "use_encoded_value"
    ordinal_unknown_value: int = -1

    @model_validator(mode="after")
    def _validate(self) -> "CategoricalFEDefaults":
        if not (0 <= self.min_bin_pct < 0.5):
            raise ValueError(
                f"min_bin_pct must satisfy 0 <= x < 0.5 (got {self.min_bin_pct})"
            )
        if self.smoothing < 0:
            raise ValueError(f"smoothing must be >= 0 (got {self.smoothing})")
        return self


class ColumnFEOverride(_Frozen):
    treat_as: Optional[Literal["numeric", "categorical"]] = None
    method: Optional[_ANY_METHOD] = None
    woe_binning: Optional[_BINNING] = None
    woe_n_bins: Optional[int] = None
    woe_min_bin_pct: Optional[float] = None
    woe_min_iv: Optional[float] = None
    minmax_range: Optional[tuple[float, float]] = None
    robust_quantile_range: Optional[tuple[float, float]] = None
    min_bin_pct: Optional[float] = None
    smoothing: Optional[float] = None
    one_hot_drop: Optional[Literal["first", "if_binary"]] = None
    ordinal_handle_unknown: Optional[Literal["use_encoded_value", "error"]] = None
    ordinal_unknown_value: Optional[int] = None

    @model_validator(mode="after")
    def _validate(self) -> "ColumnFEOverride":
        if self.woe_n_bins is not None and self.woe_n_bins < 2:
            raise ValueError(f"woe_n_bins must be >= 2 (got {self.woe_n_bins})")
        if self.woe_min_bin_pct is not None and not (0 < self.woe_min_bin_pct < 0.5):
            raise ValueError(
                f"woe_min_bin_pct must satisfy 0 < x < 0.5 (got {self.woe_min_bin_pct})"
            )
        if self.woe_min_iv is not None and self.woe_min_iv < 0:
            raise ValueError(f"woe_min_iv must be >= 0 (got {self.woe_min_iv})")
        if self.minmax_range is not None:
            lo, hi = self.minmax_range
            if not lo < hi:
                raise ValueError(
                    f"minmax_range must satisfy low < high (got {(lo, hi)})"
                )
        if self.robust_quantile_range is not None:
            qlo, qhi = self.robust_quantile_range
            if not (0 <= qlo < qhi <= 1):
                raise ValueError(
                    f"robust_quantile_range must satisfy 0 <= low < high <= 1 (got {(qlo, qhi)})"
                )
        if self.min_bin_pct is not None and not (0 <= self.min_bin_pct < 0.5):
            raise ValueError(
                f"min_bin_pct must satisfy 0 <= x < 0.5 (got {self.min_bin_pct})"
            )
        if self.smoothing is not None and self.smoothing < 0:
            raise ValueError(f"smoothing must be >= 0 (got {self.smoothing})")
        return self


class FeatureEngineeringConfig(_Frozen):
    numeric: NumericFEDefaults = Field(default_factory=NumericFEDefaults)
    categorical: CategoricalFEDefaults = Field(default_factory=CategoricalFEDefaults)
    overrides: dict[str, ColumnFEOverride] = Field(default_factory=dict)
