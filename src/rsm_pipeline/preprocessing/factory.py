"""Preprocessor factory: resolve per-column specs, build ColumnTransformer, summarize fitted state."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from rsm_pipeline.config.schema import RsmConfig
from rsm_pipeline.preprocessing.dtypes import infer_column_types
from rsm_pipeline.preprocessing.imputers.custom_fill import CustomFillImputer
from rsm_pipeline.preprocessing.imputers.knn import KNNImputerWrapper
from rsm_pipeline.preprocessing.imputers.simple import (
    MeanImputer,
    MedianImputer,
    ModeImputer,
    _AppendIsMissing,
)
from rsm_pipeline.preprocessing.outliers.iqr import IQRClipper
from rsm_pipeline.preprocessing.outliers.percentile import PercentileClipper
from rsm_pipeline.preprocessing.outliers.zscore import ZScoreCapper
from rsm_pipeline.preprocessing.schema import PreprocessingConfig

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedColSpec:
    column: str
    kind: str  # "numeric" | "categorical"
    missing: str
    add_indicator: bool
    custom_fill: Any | None
    knn_k: int
    outlier: str  # always "none" if kind == "categorical"
    iqr_factor: float
    percentile_low: float
    percentile_high: float
    zscore_threshold: float


def _resolve_column_spec(
    col: str, inferred_kind: str, cfg: PreprocessingConfig
) -> ResolvedColSpec:
    """Merge defaults + override into a single resolved spec."""
    override = cfg.overrides.get(col)
    final_kind = override.treat_as if override and override.treat_as else inferred_kind

    if final_kind == "numeric":
        base = cfg.numeric.model_dump()
    else:
        base = cfg.categorical.model_dump()
        # Categorical defaults don't carry numeric outlier fields; supply safe
        # defaults so ResolvedColSpec construction never fails.
        base.setdefault("knn_k", 5)
        base.setdefault("outlier", "none")
        base.setdefault("iqr_factor", 1.5)
        base.setdefault("percentile_low", 0.01)
        base.setdefault("percentile_high", 0.99)
        base.setdefault("zscore_threshold", 3.0)

    if override is not None:
        for field, value in override.model_dump().items():
            if value is None:
                continue
            if field == "treat_as":
                continue
            base[field] = value

    if final_kind == "categorical" and base.get("outlier", "none") != "none":
        _log.warning(
            "column %r: outlier=%r dropped because resolved kind is categorical",
            col,
            base["outlier"],
        )
        base["outlier"] = "none"

    missing = base["missing"]
    outlier = base["outlier"]
    if missing == "custom" and base.get("custom_fill") is None:
        raise ValueError(f"column {col!r}: missing=='custom' requires custom_fill")
    if missing == "mean" and final_kind != "numeric":
        raise ValueError(f"column {col!r}: missing=='mean' requires numeric dtype")
    if missing == "median" and final_kind != "numeric":
        raise ValueError(f"column {col!r}: missing=='median' requires numeric dtype")
    if missing == "knn" and final_kind != "numeric":
        raise ValueError(f"column {col!r}: missing=='knn' requires numeric dtype")
    if missing == "mode" and final_kind != "categorical":
        raise ValueError(f"column {col!r}: missing=='mode' requires categorical dtype")

    return ResolvedColSpec(
        column=col,
        kind=final_kind,
        missing=missing,
        add_indicator=bool(base.get("add_indicator", False)),
        custom_fill=base.get("custom_fill"),
        knn_k=int(base.get("knn_k", 5)),
        outlier=outlier,
        iqr_factor=float(base.get("iqr_factor", 1.5)),
        percentile_low=float(base.get("percentile_low", 0.01)),
        percentile_high=float(base.get("percentile_high", 0.99)),
        zscore_threshold=float(base.get("zscore_threshold", 3.0)),
    )


def _build_imputer(spec: ResolvedColSpec) -> BaseEstimator | None:
    m = spec.missing
    if m == "none":
        return None
    if m == "mean":
        return MeanImputer()
    if m == "median":
        return MedianImputer()
    if m == "mode":
        return ModeImputer()
    if m == "knn":
        return KNNImputerWrapper(n_neighbors=spec.knn_k)
    if m == "custom":
        return CustomFillImputer(fill_value=spec.custom_fill)
    raise ValueError(f"unknown missing strategy: {m!r}")


def _build_clipper(spec: ResolvedColSpec) -> BaseEstimator | None:
    if spec.kind != "numeric":
        return None
    o = spec.outlier
    if o == "none":
        return None
    if o == "iqr":
        return IQRClipper(factor=spec.iqr_factor)
    if o == "percentile":
        return PercentileClipper(low=spec.percentile_low, high=spec.percentile_high)
    if o == "zscore":
        return ZScoreCapper(threshold=spec.zscore_threshold)
    raise ValueError(f"unknown outlier strategy: {o!r}")


def build_preprocessor(cfg: RsmConfig, df: pd.DataFrame) -> ColumnTransformer:
    """Build an unfitted ColumnTransformer shaped from ``df.dtypes`` and ``cfg.preprocessing``.

    Caller must drop target / date columns from ``df`` before calling.
    Returned object is unfitted; fit on training data.
    """
    if cfg.preprocessing is None:
        raise ValueError("build_preprocessor called but cfg.preprocessing is None")

    pre_cfg = cfg.preprocessing
    types = infer_column_types(df)

    unknown = [c for c in pre_cfg.overrides.keys() if c not in types]
    if unknown:
        raise ValueError(
            f"preprocessing.overrides references unknown columns: {unknown}"
        )

    transformers: list[tuple[str, Any, list[str]]] = []
    for col, kind in types.items():
        spec = _resolve_column_spec(col, kind, pre_cfg)
        steps: list[tuple[str, BaseEstimator]] = []
        if spec.add_indicator:
            steps.append(("indicator", _AppendIsMissing()))
        imp = _build_imputer(spec)
        if imp is not None:
            steps.append(("impute", imp))
        clip = _build_clipper(spec)
        if clip is not None:
            steps.append(("clip", clip))

        if not steps:
            transformers.append((f"col_{col}", "passthrough", [col]))
        else:
            transformers.append((f"col_{col}", Pipeline(steps), [col]))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _jsonify(v: Any, infinite_as: str = "null") -> Any:
    """Convert numpy scalars / arrays into JSON-friendly values, rounded to 6 dp.

    ``infinite_as`` controls how non-finite floats are serialized:
    - "null" (default, preprocessing behavior): returns None for inf/-inf/NaN
    - "string" (WoE summarization): returns "+inf" / "-inf" / "nan"
    """
    if isinstance(v, (np.floating, float)):
        if not np.isfinite(v):
            if infinite_as == "string":
                if np.isnan(v):
                    return "nan"
                return "+inf" if v > 0 else "-inf"
            return None
        return round(float(v), 6)
    if isinstance(v, (np.integer, int)) and not isinstance(v, bool):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, np.ndarray):
        return [_jsonify(x, infinite_as=infinite_as) for x in v.tolist()]
    return v


def _summarize_step(step_name: str, transformer: BaseEstimator) -> dict[str, Any]:
    """Pull JSON-friendly fitted state out of a single fitted transformer."""
    out: dict[str, Any] = {}
    cls = transformer.__class__.__name__
    if cls in ("MeanImputer", "MedianImputer", "ModeImputer"):
        key = {"MeanImputer": "mean", "MedianImputer": "median", "ModeImputer": "mode"}[
            cls
        ]
        out[key] = _jsonify(transformer.statistics_[0])
    elif cls == "KNNImputerWrapper":
        out["knn_k"] = int(transformer.n_neighbors)
    elif cls == "CustomFillImputer":
        out["fill_value"] = transformer.fill_value
    elif cls == "IQRClipper":
        out["iqr_lower"] = _jsonify(transformer.lower_[0])
        out["iqr_upper"] = _jsonify(transformer.upper_[0])
        out["iqr_factor"] = float(transformer.factor)
    elif cls == "PercentileClipper":
        out["pct_lower"] = _jsonify(transformer.lower_[0])
        out["pct_upper"] = _jsonify(transformer.upper_[0])
        out["low"] = float(transformer.low)
        out["high"] = float(transformer.high)
    elif cls == "ZScoreCapper":
        out["mean"] = _jsonify(transformer.mean_[0])
        out["std"] = _jsonify(transformer.std_[0])
        out["zscore_lower"] = _jsonify(transformer.lower_[0])
        out["zscore_upper"] = _jsonify(transformer.upper_[0])
    return out


def _summarize_preprocessor(
    pre: ColumnTransformer, types: dict[str, str], pre_cfg: PreprocessingConfig
) -> dict[str, Any]:
    """Produce a JSON-serializable summary of a fitted ColumnTransformer."""
    columns: dict[str, Any] = {}
    n_input_cols = len(types)
    n_output_cols = len(list(pre.get_feature_names_out()))

    for name, fitted, col_list in pre.transformers_:
        if name == "remainder":
            continue
        if not col_list:
            continue
        col = col_list[0]
        spec = _resolve_column_spec(col, types[col], pre_cfg)
        fitted_state: dict[str, Any] = {}
        if hasattr(fitted, "named_steps"):
            for step_name, step in fitted.named_steps.items():
                if step_name == "indicator":
                    continue
                fitted_state.update(_summarize_step(step_name, step))
        columns[col] = {
            "kind": spec.kind,
            "missing": spec.missing,
            "add_indicator": spec.add_indicator,
            "outlier": spec.outlier,
            "fitted": fitted_state,
        }
        if spec.outlier == "iqr":
            columns[col]["iqr_factor"] = spec.iqr_factor
        if spec.outlier == "percentile":
            columns[col]["percentile_low"] = spec.percentile_low
            columns[col]["percentile_high"] = spec.percentile_high
        if spec.outlier == "zscore":
            columns[col]["zscore_threshold"] = spec.zscore_threshold
        if spec.missing == "knn":
            columns[col]["knn_k"] = spec.knn_k
        if spec.missing == "custom":
            columns[col]["custom_fill"] = spec.custom_fill

    return {
        "applied": True,
        "n_input_cols": n_input_cols,
        "n_output_cols": n_output_cols,
        "columns": columns,
    }
