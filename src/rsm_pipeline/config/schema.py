"""Pydantic v2 models for the pipeline configuration.

All models are frozen (``ConfigDict(frozen=True)``) and reject unknown
fields (``extra="forbid"``). Path fields are typed as ``str`` here;
conversion to ``Path`` happens in :mod:`rsm_pipeline.config.loader`.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal, Optional

from pydantic import Field, model_validator

from rsm_pipeline._frozen import _Frozen  # noqa: F401  (re-exported for back-compat)


# --- run -------------------------------------------------------------


class RunConfig(_Frozen):
    name: str
    seed: int = 42
    output_root: str = "experiments"


# --- data ------------------------------------------------------------


class CsvOptions(_Frozen):
    sep: str = ","
    encoding: str = "utf-8"


class DataSourceConfig(_Frozen):
    format: Literal["csv", "parquet"]
    path: str
    csv_options: CsvOptions = Field(default_factory=CsvOptions)


class TargetConfig(_Frozen):
    column: str
    positive_class: Any = 1


class DataConfig(_Frozen):
    source: DataSourceConfig
    target: TargetConfig
    date_column: Optional[str] = None


# --- split -----------------------------------------------------------


def _check_ratio_sum(train: float, val: float, test: float) -> None:
    total = train + val + test
    if abs(total - 1.0) > 1e-3:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must sum to 1.0 " f"(got {total!r})"
        )


class _RatioBlock(_Frozen):
    train_ratio: float
    val_ratio: float
    test_ratio: float

    @model_validator(mode="after")
    def _ratios_sum_to_one(self) -> "_RatioBlock":
        _check_ratio_sum(self.train_ratio, self.val_ratio, self.test_ratio)
        return self


class RandomSplitConfig(_RatioBlock):
    pass


class StratifiedRandomSplitConfig(_RatioBlock):
    stratify_by: Optional[str] = None


class TimeRatioSplitConfig(_RatioBlock):
    pass


class TimeCutoffSplitConfig(_Frozen):
    train_end: date
    val_end: date

    @model_validator(mode="after")
    def _ordered(self) -> "TimeCutoffSplitConfig":
        if not (self.train_end < self.val_end):
            raise ValueError(
                f"train_end ({self.train_end}) must be < val_end ({self.val_end})"
            )
        return self


class TimeSplitConfig(_Frozen):
    date_column: Optional[str] = None
    mode: Literal["ratio", "cutoff"] = "ratio"
    ratio: Optional[TimeRatioSplitConfig] = None
    cutoff: Optional[TimeCutoffSplitConfig] = None

    @model_validator(mode="after")
    def _block_for_mode_present(self) -> "TimeSplitConfig":
        if self.mode == "ratio" and self.ratio is None:
            raise ValueError("split.time.mode == 'ratio' requires split.time.ratio")
        if self.mode == "cutoff" and self.cutoff is None:
            raise ValueError("split.time.mode == 'cutoff' requires split.time.cutoff")
        return self


class SplitConfig(_Frozen):
    method: Literal["random", "stratified_random", "time"]
    random: Optional[RandomSplitConfig] = None
    stratified_random: Optional[StratifiedRandomSplitConfig] = None
    time: Optional[TimeSplitConfig] = None

    @model_validator(mode="after")
    def _block_for_method_present(self) -> "SplitConfig":
        if self.method == "random" and self.random is None:
            raise ValueError("split.method == 'random' requires split.random")
        if self.method == "stratified_random" and self.stratified_random is None:
            raise ValueError(
                "split.method == 'stratified_random' requires "
                "split.stratified_random"
            )
        if self.method == "time" and self.time is None:
            raise ValueError("split.method == 'time' requires split.time")
        return self


# --- top-level -------------------------------------------------------


class RsmConfig(_Frozen):
    run: RunConfig
    data: DataConfig
    split: SplitConfig
    preprocessing: Optional["PreprocessingConfig"] = None
    feature_engineering: Optional["FeatureEngineeringConfig"] = None
    feature_selection: Optional["FeatureSelectionConfig"] = None
    imbalance: Optional["ImbalanceConfig"] = None
    tuning: Optional["TuningConfig"] = None
    model: "ModelConfig"
    evaluation: "EvaluationConfig"
    calibration: Optional["CalibrationConfig"] = None
    scorecard: Optional["ScorecardConfig"] = None
    explain: Optional["ExplainConfig"] = None
    export: Optional["ExportConfig"] = None

    @model_validator(mode="after")
    def _fe_requires_preprocessing(self) -> "RsmConfig":
        if self.feature_engineering is not None and self.preprocessing is None:
            raise ValueError(
                "feature_engineering requires preprocessing to be configured "
                "(otherwise NaN would reach encoders/scalers)"
            )
        return self

    @model_validator(mode="after")
    def _date_column_present_when_time_split(self) -> "RsmConfig":
        if self.split.method != "time":
            return self
        time_col = (self.split.time.date_column if self.split.time else None) or (
            self.data.date_column
        )
        if not time_col:
            raise ValueError(
                "split.method == 'time' requires either data.date_column or "
                "split.time.date_column to be set"
            )
        if (
            self.split.time
            and self.split.time.date_column
            and self.data.date_column
            and self.split.time.date_column != self.data.date_column
        ):
            raise ValueError(
                "data.date_column and split.time.date_column must match "
                f"({self.data.date_column!r} vs {self.split.time.date_column!r})"
            )
        return self


# Resolve forward reference: PreprocessingConfig is defined in a sibling module
# that itself imports _Frozen from this file, so we cannot import it at the top
# without creating a cycle. Calling model_rebuild() here defers the resolution
# until both modules are fully loaded.
def _rebuild_forward_refs() -> None:
    from rsm_pipeline.preprocessing.schema import PreprocessingConfig  # noqa: F401
    from rsm_pipeline.feature_engineering.schema import (  # noqa: F401
        FeatureEngineeringConfig,
    )
    from rsm_pipeline.feature_selection.schema import (  # noqa: F401
        FeatureSelectionConfig,
    )
    from rsm_pipeline.imbalance.schema import ImbalanceConfig  # noqa: F401
    from rsm_pipeline.models.schema import ModelConfig  # noqa: F401
    from rsm_pipeline.tuning.schema import TuningConfig  # noqa: F401
    from rsm_pipeline.evaluation.schema import EvaluationConfig  # noqa: F401
    from rsm_pipeline.calibration.schema import CalibrationConfig  # noqa: F401
    from rsm_pipeline.scorecard.schema import ScorecardConfig  # noqa: F401
    from rsm_pipeline.explain.schema import ExplainConfig  # noqa: F401
    from rsm_pipeline.export.schema import ExportConfig  # noqa: F401

    RsmConfig.model_rebuild()


_rebuild_forward_refs()
