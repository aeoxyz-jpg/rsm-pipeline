"""Feature engineering factory: resolve specs, build ColumnTransformer, summarize."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer

from rsm_pipeline.config.schema import RsmConfig
from rsm_pipeline.feature_engineering.encoders.one_hot import OneHotEncoderWrapper
from rsm_pipeline.feature_engineering.encoders.ordinal import OrdinalEncoderWrapper
from rsm_pipeline.feature_engineering.encoders.target import TargetEncoderWrapper
from rsm_pipeline.feature_engineering.scalers.minmax import MinMaxScalerWrapper
from rsm_pipeline.feature_engineering.scalers.robust import RobustScalerWrapper
from rsm_pipeline.feature_engineering.scalers.standard import StandardScalerWrapper
from rsm_pipeline.feature_engineering.schema import FeatureEngineeringConfig
from rsm_pipeline.feature_engineering.woe.categorical import CategoricalWoEEncoder
from rsm_pipeline.feature_engineering.woe.numerical import NumericalWoEEncoder
from rsm_pipeline.preprocessing.dtypes import infer_column_types
from rsm_pipeline.preprocessing.factory import _jsonify

_log = logging.getLogger(__name__)

_NUMERIC_METHODS = {"standard", "minmax", "robust", "woe", "none"}
_CATEGORICAL_METHODS = {"woe", "one_hot", "ordinal", "target", "none"}


@dataclass(frozen=True)
class ResolvedFESpec:
    column: str
    kind: str
    method: str
    woe_binning: str
    woe_n_bins: int
    woe_min_bin_pct: float
    woe_min_iv: float
    minmax_low: float
    minmax_high: float
    robust_q_low: float
    robust_q_high: float
    min_bin_pct: float
    smoothing: float
    one_hot_drop: Optional[str]
    ordinal_handle_unknown: str
    ordinal_unknown_value: int


def _resolve_fe_spec(
    col: str, inferred_kind: str, cfg: FeatureEngineeringConfig
) -> ResolvedFESpec:
    override = cfg.overrides.get(col)
    final_kind = override.treat_as if override and override.treat_as else inferred_kind

    if final_kind == "numeric":
        base = cfg.numeric.model_dump()
        # Categorical-only fields with safe defaults so dataclass build works.
        base.setdefault("min_bin_pct", 0.01)
        base.setdefault("smoothing", 0.0)
        base.setdefault("one_hot_drop", None)
        base.setdefault("ordinal_handle_unknown", "use_encoded_value")
        base.setdefault("ordinal_unknown_value", -1)
    else:
        base = cfg.categorical.model_dump()
        base.setdefault("woe_binning", "quantile")
        base.setdefault("woe_n_bins", 10)
        base.setdefault("woe_min_bin_pct", 0.05)
        base.setdefault("woe_min_iv", 0.0)
        base.setdefault("minmax_range", (0.0, 1.0))
        base.setdefault("robust_quantile_range", (0.25, 0.75))

    if override is not None:
        for field, value in override.model_dump().items():
            if value is None or field == "treat_as":
                continue
            base[field] = value

    method = base["method"]
    if final_kind == "numeric" and method not in _NUMERIC_METHODS:
        raise ValueError(f"column {col!r}: method {method!r} requires numeric kind")
    if final_kind == "categorical" and method not in _CATEGORICAL_METHODS:
        raise ValueError(f"column {col!r}: method {method!r} requires categorical kind")

    minmax_lo, minmax_hi = base["minmax_range"]
    robust_qlo, robust_qhi = base["robust_quantile_range"]

    return ResolvedFESpec(
        column=col,
        kind=final_kind,
        method=method,
        woe_binning=str(base["woe_binning"]),
        woe_n_bins=int(base["woe_n_bins"]),
        woe_min_bin_pct=float(base["woe_min_bin_pct"]),
        woe_min_iv=float(base["woe_min_iv"]),
        minmax_low=float(minmax_lo),
        minmax_high=float(minmax_hi),
        robust_q_low=float(robust_qlo),
        robust_q_high=float(robust_qhi),
        min_bin_pct=float(base["min_bin_pct"]),
        smoothing=float(base["smoothing"]),
        one_hot_drop=base["one_hot_drop"],
        ordinal_handle_unknown=str(base["ordinal_handle_unknown"]),
        ordinal_unknown_value=int(base["ordinal_unknown_value"]),
    )


def _build_transformer(spec: ResolvedFESpec) -> Optional[BaseEstimator]:
    if spec.method == "none":
        return None
    if spec.kind == "numeric":
        if spec.method == "standard":
            return StandardScalerWrapper()
        if spec.method == "minmax":
            return MinMaxScalerWrapper(low=spec.minmax_low, high=spec.minmax_high)
        if spec.method == "robust":
            return RobustScalerWrapper(
                q_low=spec.robust_q_low, q_high=spec.robust_q_high
            )
        if spec.method == "woe":
            return NumericalWoEEncoder(
                binning=spec.woe_binning,
                n_bins=spec.woe_n_bins,
                min_bin_pct=spec.woe_min_bin_pct,
            )
    else:
        if spec.method == "woe":
            return CategoricalWoEEncoder(min_bin_pct=spec.min_bin_pct)
        if spec.method == "one_hot":
            return OneHotEncoderWrapper(drop=spec.one_hot_drop)
        if spec.method == "ordinal":
            return OrdinalEncoderWrapper(
                handle_unknown=spec.ordinal_handle_unknown,
                unknown_value=spec.ordinal_unknown_value,
            )
        if spec.method == "target":
            return TargetEncoderWrapper(smoothing=spec.smoothing)
    raise ValueError(f"unbuildable spec: {spec}")


def build_feature_engineer(cfg: RsmConfig, df: pd.DataFrame) -> ColumnTransformer:
    if cfg.feature_engineering is None:
        raise ValueError(
            "build_feature_engineer called but cfg.feature_engineering is None"
        )

    fe_cfg = cfg.feature_engineering
    types = infer_column_types(df)

    unknown = [c for c in fe_cfg.overrides.keys() if c not in types]
    if unknown:
        raise ValueError(
            f"feature_engineering.overrides references unknown columns: {unknown}"
        )

    transformers: list[tuple[str, Any, list[str]]] = []
    for col, kind in types.items():
        spec = _resolve_fe_spec(col, kind, fe_cfg)
        t = _build_transformer(spec)
        if t is None:
            transformers.append((f"col_{col}", "passthrough", [col]))
        else:
            transformers.append((f"col_{col}", t, [col]))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _summarize_fitted(spec: ResolvedFESpec, t: BaseEstimator) -> dict[str, Any]:
    out: dict[str, Any] = {}
    cls = t.__class__.__name__
    if cls == "StandardScalerWrapper":
        out["mean"] = _jsonify(t.mean_[0])
        out["scale"] = _jsonify(t.scale_[0])
    elif cls == "MinMaxScalerWrapper":
        out["data_min"] = _jsonify(t.data_min_[0])
        out["data_max"] = _jsonify(t.data_max_[0])
        out["range_low"] = float(t.low)
        out["range_high"] = float(t.high)
    elif cls == "RobustScalerWrapper":
        out["center"] = _jsonify(t.center_[0])
        out["scale"] = _jsonify(t.scale_[0])
    elif cls == "OneHotEncoderWrapper":
        out["categories"] = [[_jsonify(x) for x in cat] for cat in t.categories_]
        out["n_output_cols"] = len(t._out_names)
    elif cls == "OrdinalEncoderWrapper":
        out["categories"] = [[_jsonify(x) for x in cat] for cat in t.categories_]
    elif cls == "TargetEncoderWrapper":
        out["global_mean"] = _jsonify(t.global_mean_)
        out["n_categories"] = len(t.encodings_)
        out["smoothing"] = float(t.smoothing)
    elif cls in ("NumericalWoEEncoder", "CategoricalWoEEncoder"):
        table = t.table_
        bins_out = [
            {
                "label": b.label,
                "n_total": int(b.n_total),
                "n_pos": int(b.n_pos),
                "n_neg": int(b.n_neg),
                "pos_rate": _jsonify(b.pos_rate),
                "woe": _jsonify(b.woe),
                "iv_contrib": _jsonify(b.iv_contrib),
            }
            for b in table.bins
        ]
        out["iv"] = _jsonify(table.iv)
        out["bins"] = bins_out
        if cls == "NumericalWoEEncoder":
            out["cuts"] = [_jsonify(c, infinite_as="string") for c in t.cuts_]
            out["nan_woe"] = _jsonify(t.nan_woe_)
            out["n_bins_actual"] = len(t.cuts_) - 1
        else:
            out["rare_woe"] = _jsonify(t.rare_woe_)
    return out


def _summarize_feature_engineer(
    fe: ColumnTransformer,
    types: dict[str, str],
    fe_cfg: FeatureEngineeringConfig,
) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    n_input_cols = len(types)
    n_output_cols = len(list(fe.get_feature_names_out()))

    for name, fitted, col_list in fe.transformers_:
        if name == "remainder":
            continue
        if not col_list:
            continue
        col = col_list[0]
        spec = _resolve_fe_spec(col, types[col], fe_cfg)

        col_summary: dict[str, Any] = {
            "kind": spec.kind,
            "method": spec.method,
            "fitted": {},
        }
        if spec.method == "woe":
            col_summary["woe_binning"] = spec.woe_binning
            col_summary["woe_n_bins"] = spec.woe_n_bins
            col_summary["woe_min_bin_pct"] = spec.woe_min_bin_pct
            col_summary["woe_min_iv"] = spec.woe_min_iv
        if spec.method == "minmax":
            col_summary["minmax_range"] = [spec.minmax_low, spec.minmax_high]
        if spec.method == "robust":
            col_summary["robust_quantile_range"] = [
                spec.robust_q_low,
                spec.robust_q_high,
            ]
        if spec.method == "target":
            col_summary["smoothing"] = spec.smoothing
        if spec.method == "one_hot":
            col_summary["one_hot_drop"] = spec.one_hot_drop
        if spec.method == "ordinal":
            col_summary["ordinal_unknown_value"] = spec.ordinal_unknown_value

        if isinstance(fitted, str) and fitted == "passthrough":
            pass  # method=='none', nothing fitted
        else:
            col_summary["fitted"] = _summarize_fitted(spec, fitted)

        columns[col] = col_summary

    return {
        "applied": True,
        "n_input_cols": n_input_cols,
        "n_output_cols": n_output_cols,
        "columns": columns,
    }
