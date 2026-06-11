"""apply_export: build TrainedBundle, joblib + optional ONNX/PMML."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import joblib

from rsm_pipeline.config.schema import RsmConfig
from rsm_pipeline.data.splitter import SplitResult
from rsm_pipeline.export.bundle import TrainedBundle, _build_meta
from rsm_pipeline.export.onnx_export import _export_onnx
from rsm_pipeline.export.pmml_export import _export_pmml

_log = logging.getLogger(__name__)


def apply_export(
    model: Any,
    scorer: Optional[Any],
    preprocessing: Optional[Any],
    feature_engineering: Optional[Any],
    feature_selection: Optional[Any],
    sp: SplitResult,
    cfg: RsmConfig,
    feats_after_fs: list[str],
    raw_input_columns: list[str],
    target: str,
    run_dir: Path,
) -> dict[str, Any]:
    assert cfg.export is not None
    ex_cfg = cfg.export
    bundle = TrainedBundle(
        preprocessing=preprocessing,
        feature_engineering=feature_engineering,
        feature_selection=feature_selection,
        model=model,
        scorer=scorer,
        feats_after_fs=list(feats_after_fs),
        target=target,
        date_column=cfg.data.date_column,
        raw_input_columns=list(raw_input_columns),
    )
    bundle_path = run_dir / ex_cfg.bundle_path
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, bundle_path)

    onnx_status: dict[str, Any] = {"status": "not_requested"}
    if "onnx" in ex_cfg.kinds:
        onnx_status = _export_onnx(
            model, feature_engineering, feats_after_fs, run_dir / ex_cfg.onnx_path
        )

    pmml_status: dict[str, Any] = {"status": "not_requested"}
    if "pmml" in ex_cfg.kinds:
        pmml_status = _export_pmml(
            model,
            preprocessing,
            feature_engineering,
            feature_selection,
            run_dir / ex_cfg.pmml_path,
        )

    summary = {
        "kinds": list(ex_cfg.kinds),
        "bundle_path": ex_cfg.bundle_path,
        "onnx": onnx_status,
        "pmml": pmml_status,
        "build_meta": _build_meta(),
    }
    summary_path = run_dir / ex_cfg.summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary
