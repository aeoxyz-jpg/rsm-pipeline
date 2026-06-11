"""apply_monitoring orchestrator: PSI + CSI + quality + drift + summary."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from rsm_pipeline.monitoring.csi import _compute_csi_per_feature
from rsm_pipeline.monitoring.drift import _performance_drift
from rsm_pipeline.monitoring.psi import _compute_psi, _tier
from rsm_pipeline.monitoring.quality import _data_quality_checks
from rsm_pipeline.serving.validate import _validate_input_columns

_log = logging.getLogger(__name__)


def apply_monitoring(
    bundle: Any,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_dir: Path,
    *,
    reference_metrics: Optional[dict] = None,
    has_labels: bool = False,
    target: Optional[str] = None,
    n_bins: int = 10,
    minor_threshold: float = 0.1,
    major_threshold: float = 0.25,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate columns once for each (drop extras, check missing)
    ref_clean, _ = _validate_input_columns(reference, bundle)
    cur_clean, _ = _validate_input_columns(current, bundle)

    # 1. Score PSI
    ref_score = bundle.predict_proba(ref_clean)[:, 1]
    cur_score = bundle.predict_proba(cur_clean)[:, 1]
    psi_value, psi_breakdown = _compute_psi(ref_score, cur_score, n_bins=n_bins)
    psi_block = {
        "value": psi_value,
        "tier": _tier(psi_value, minor_threshold, major_threshold),
        "n_bins": n_bins,
        "breakdown": psi_breakdown,
    }

    # 2. CSI per feature
    csi_rows = _compute_csi_per_feature(
        ref_clean,
        cur_clean,
        list(bundle.raw_input_columns),
        n_bins=n_bins,
        minor=minor_threshold,
        major=major_threshold,
    )
    csi_df = pd.DataFrame(csi_rows)
    csi_df.to_csv(output_dir / "csi_per_feature.csv", encoding="utf-8", index=False)
    n_major = int((csi_df["tier"] == "major_shift").sum())
    n_minor = int((csi_df["tier"] == "minor_shift").sum())
    if n_major > 0 or n_minor > 0:
        worst_idx = (
            csi_df.dropna(subset=["csi"])["csi"].astype(float).idxmax()
            if csi_df["csi"].notna().any()
            else None
        )
    else:
        worst_idx = (
            csi_df["csi"].astype(float).idxmax()
            if csi_df["csi"].notna().any()
            else None
        )
    csi_block = {
        "max_value": (
            float(csi_df.loc[worst_idx, "csi"]) if worst_idx is not None else None
        ),
        "max_feature": (
            str(csi_df.loc[worst_idx, "feature"]) if worst_idx is not None else None
        ),
        "tier": (
            str(csi_df.loc[worst_idx, "tier"]) if worst_idx is not None else "stable"
        ),
        "n_major": n_major,
        "n_minor": n_minor,
    }

    # 3. Data quality
    quality = _data_quality_checks(reference, current, list(bundle.raw_input_columns))
    (output_dir / "data_quality.json").write_text(
        json.dumps(quality, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    worst_missing_feat: Optional[str] = None
    worst_missing_val = 0.0
    ks_significant = 0
    features_with_new_levels: list[str] = []
    for col, info in quality.items():
        delta = info["missing_rate_delta"]
        if abs(delta) > abs(worst_missing_val):
            worst_missing_val = float(delta)
            worst_missing_feat = col
        if info.get("ks_p_value") is not None and info["ks_p_value"] < 0.05:
            ks_significant += 1
        if info.get("new_levels"):
            features_with_new_levels.append(col)
    quality_block = {
        "worst_missing_delta": {
            "feature": worst_missing_feat,
            "value": worst_missing_val,
        },
        "ks_significant_count": ks_significant,
        "features_with_new_levels": features_with_new_levels,
    }

    # 4. Performance drift
    perf_block: Optional[dict[str, Any]] = None
    if has_labels:
        target_col = target or bundle.target
        if target_col not in current.columns:
            raise ValueError(
                f"--has-labels but target {target_col!r} not in current data"
            )
        # Build a frame containing input columns + target for drift
        cur_with_label = current[
            [c for c in bundle.raw_input_columns] + [target_col]
        ].copy()
        # Rename to match bundle's expected target name for drift function
        cur_with_label = cur_with_label.rename(columns={target_col: "_y"})
        # Bundle.predict_proba runs on raw_input_columns only; pass target separately
        df = cur_with_label.dropna(subset=["_y"]).copy()
        y_true = df["_y"].astype(int)
        proba = bundle.predict_proba(df.drop(columns=["_y"]))
        y_score = proba[:, 1]
        from rsm_pipeline.evaluation.decile import _compute_decile_lift
        from rsm_pipeline.evaluation.metrics import _compute_scalar_metrics

        metrics = _compute_scalar_metrics(y_true, y_score, thresholds=[0.5])
        decile = _compute_decile_lift(y_true, y_score)
        decile.to_csv(
            output_dir / "decile_lift_current.csv",
            encoding="utf-8",
            index=False,
        )
        keys = ("roc_auc", "ks", "gini", "pr_auc", "brier", "log_loss")
        summary: dict[str, Any] = {
            "current_metrics": {k: metrics[k] for k in keys},
            "n_current_with_labels": int(len(df)),
        }
        if reference_metrics:
            deltas: dict[str, float] = {}
            for k in keys:
                ref_v = reference_metrics.get(k)
                if ref_v is None:
                    continue
                deltas[k] = float(metrics[k] - ref_v)
            summary["reference_metrics"] = {k: reference_metrics.get(k) for k in keys}
            summary["deltas"] = deltas
            d = deltas.get("roc_auc")
            if d is None:
                summary["tier"] = "no_reference_roc_auc"
            elif d <= -0.05:
                summary["tier"] = "major_drop"
            elif d <= -0.01:
                summary["tier"] = "minor_drop"
            else:
                summary["tier"] = "stable"
        else:
            summary["tier"] = "no_reference"
        (output_dir / "performance_drift.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        perf_block = {
            "present": True,
            "tier": summary["tier"],
            "delta_roc_auc": (summary.get("deltas") or {}).get("roc_auc"),
            "delta_brier": (summary.get("deltas") or {}).get("brier"),
        }

    monitor_summary = {
        "score_psi": {
            "value": psi_block["value"],
            "tier": psi_block["tier"],
            "n_bins": psi_block["n_bins"],
        },
        "csi": csi_block,
        "data_quality": quality_block,
        "performance": perf_block,
        "n_reference": int(len(reference)),
        "n_current": int(len(current)),
    }
    (output_dir / "monitor_summary.json").write_text(
        json.dumps(monitor_summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    # Also dump full PSI breakdown for diagnostics
    (output_dir / "score_psi_breakdown.json").write_text(
        json.dumps(psi_block, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return monitor_summary
