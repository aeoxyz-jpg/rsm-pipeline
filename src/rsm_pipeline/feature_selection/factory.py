"""Build chained selector Pipeline + summarize fitted state + write CSV report."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from rsm_pipeline.config.schema import RsmConfig
from rsm_pipeline.feature_selection.selectors._base import _PassthroughSelector
from rsm_pipeline.feature_selection.selectors.correlation import CorrelationSelector
from rsm_pipeline.feature_selection.selectors.importance import ModelImportanceSelector
from rsm_pipeline.feature_selection.selectors.iv_threshold import IVThresholdSelector
from rsm_pipeline.feature_selection.selectors.rfe import RFESelector
from rsm_pipeline.feature_selection.selectors.variance import VarianceSelector
from rsm_pipeline.preprocessing.factory import _jsonify

_log = logging.getLogger(__name__)


def build_feature_selector(cfg: RsmConfig) -> Pipeline:
    """Build an unfitted ``sklearn.pipeline.Pipeline`` of selectors."""
    if cfg.feature_selection is None:
        raise ValueError(
            "build_feature_selector called but cfg.feature_selection is None"
        )

    seed = cfg.run.seed
    fs_cfg = cfg.feature_selection
    steps: list[tuple[str, BaseEstimator]] = []
    for i, sel_cfg in enumerate(fs_cfg.selectors):
        kind = sel_cfg.kind
        name = f"{i:02d}_{kind}"
        if kind == "variance":
            sel: BaseEstimator = VarianceSelector(threshold=sel_cfg.threshold)
        elif kind == "iv_threshold":
            sel = IVThresholdSelector(
                threshold=sel_cfg.threshold,
                n_bins=sel_cfg.n_bins,
                smoothing=sel_cfg.smoothing,
            )
        elif kind == "correlation":
            sel = CorrelationSelector(
                threshold=sel_cfg.threshold,
                method=sel_cfg.method,
                tie_break=sel_cfg.tie_break,
            )
        elif kind == "importance":
            rs = sel_cfg.random_state if sel_cfg.random_state is not None else seed
            sel = ModelImportanceSelector(
                top_k=sel_cfg.top_k,
                top_k_pct=sel_cfg.top_k_pct,
                n_estimators=sel_cfg.n_estimators,
                max_depth=sel_cfg.max_depth,
                random_state=rs,
            )
        elif kind == "rfe":
            sel = RFESelector(
                n_features_to_select=sel_cfg.n_features_to_select,
                step=sel_cfg.step,
            )
        else:
            raise ValueError(f"unknown selector kind: {kind!r}")
        steps.append((name, sel))

    if not steps:
        # sklearn Pipeline cannot be empty. Insert a no-op passthrough selector.
        steps = [("00_passthrough", _PassthroughSelector())]

    return Pipeline(steps)


def _selector_stats(name: str, sel: BaseEstimator) -> dict[str, Any]:
    """JSON-friendly stats for one fitted selector."""
    cls = sel.__class__.__name__
    out: dict[str, Any] = {}
    if cls == "VarianceSelector":
        out["params"] = {"threshold": float(sel.threshold)}
        out["stats"] = {
            col: {"variance": _jsonify(var)}
            for col, var in zip(sel.feature_names_in_, sel.variances_)
        }
    elif cls == "IVThresholdSelector":
        out["params"] = {
            "threshold": float(sel.threshold),
            "n_bins": int(sel.n_bins),
            "smoothing": float(sel.smoothing),
        }
        out["stats"] = {col: {"iv": _jsonify(iv)} for col, iv in sel.ivs_.items()}
    elif cls == "CorrelationSelector":
        out["params"] = {
            "threshold": float(sel.threshold),
            "method": sel.method,
            "tie_break": sel.tie_break,
        }
        out["stats"] = {
            "dropped_pairs": [
                {"loser": l, "winner": w, "abs_corr": _jsonify(v)}
                for l, w, v in sel.dropped_pairs_
            ]
        }
    elif cls == "ModelImportanceSelector":
        out["params"] = {
            "top_k": sel.top_k,
            "top_k_pct": sel.top_k_pct,
            "n_estimators": int(sel.n_estimators),
            "max_depth": sel.max_depth,
            "random_state": sel.random_state,
        }
        out["stats"] = {
            col: {"importance": _jsonify(imp)} for col, imp in sel.importances_.items()
        }
    elif cls == "RFESelector":
        out["params"] = {
            "n_features_to_select": int(sel.n_features_to_select),
            "step": int(sel.step),
        }
        out["stats"] = {col: {"rank": int(rank)} for col, rank in sel.ranking_.items()}
    return out


def _write_report_csv(pipe: Pipeline, csv_path: Path) -> None:
    """One row per (input column x selector). Columns: column, selector, reason, kept."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["column", "selector", "reason", "kept"])
        for name, sel in pipe.steps:
            kept = set(sel.kept_columns_)
            reasons = sel.drop_reasons()
            for col in sel.feature_names_in_:
                if col in kept:
                    writer.writerow([col, name, "(passed)", "True"])
                else:
                    writer.writerow([col, name, reasons.get(col, "dropped"), "False"])


def _summarize_feature_selection(
    pipe: Pipeline, original_feats: list[str], report_csv_rel: str
) -> dict[str, Any]:
    selectors_summary: list[dict[str, Any]] = []
    dropped_total: list[dict[str, str]] = []
    for name, sel in pipe.steps:
        n_in = len(sel.feature_names_in_)
        n_out = len(sel.kept_columns_)
        n_drop = len(sel.dropped_columns_)
        stats = _selector_stats(name, sel)
        selectors_summary.append(
            {
                "name": name,
                "kind": sel.__class__.__name__,
                "n_input": n_in,
                "n_output": n_out,
                "n_dropped": n_drop,
                **stats,
            }
        )
        reasons = sel.drop_reasons()
        for col in sel.dropped_columns_:
            dropped_total.append(
                {
                    "column": col,
                    "selector": name,
                    "reason": reasons.get(col, "dropped"),
                }
            )

    final_kept = (
        list(pipe.steps[-1][1].kept_columns_) if pipe.steps else list(original_feats)
    )

    return {
        "applied": True,
        "n_input_cols": len(original_feats),
        "n_output_cols": len(final_kept),
        "selectors": selectors_summary,
        "dropped_columns_total": dropped_total,
        "report_csv": report_csv_rel,
    }
