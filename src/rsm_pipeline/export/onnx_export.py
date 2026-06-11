"""Best-effort ONNX export for non-WoE LR/RF/ET pipelines."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

_log = logging.getLogger(__name__)


def _has_woe(fe_pipeline: Any) -> bool:
    if fe_pipeline is None:
        return False
    try:
        from rsm_pipeline.feature_engineering.woe.categorical import (
            CategoricalWoEEncoder,
        )
        from rsm_pipeline.feature_engineering.woe.numerical import (
            NumericalWoEEncoder,
        )
    except ImportError:
        return False
    transformers = getattr(fe_pipeline, "transformers_", None)
    if transformers is None:
        return False
    for _name, transformer, _cols in transformers:
        if isinstance(transformer, (NumericalWoEEncoder, CategoricalWoEEncoder)):
            return True
    return False


def _is_eligible(model: Any, fe_pipeline: Any) -> tuple[bool, str]:
    from rsm_pipeline.explain.unwrap import _unwrap_to_base_model

    if _has_woe(fe_pipeline):
        return False, "WoE encoders are not supported by skl2onnx"
    base = _unwrap_to_base_model(model)
    cls = type(base).__name__
    if cls in {
        "LogisticRegression",
        "RandomForestClassifier",
        "ExtraTreesClassifier",
    }:
        return True, "skl2onnx-sklearn"
    if cls in {"XGBClassifier", "LGBMClassifier"}:
        return True, "onnxmltools"
    return False, f"no ONNX converter for {cls}"


def _export_onnx(
    model: Any,
    fe_pipeline: Any,
    feats: list[str],
    save_path: Path,
) -> dict[str, Any]:
    """Try ONNX conversion; return status dict with status/reason/path."""
    eligible, reason = _is_eligible(model, fe_pipeline)
    if not eligible:
        return {"status": "skipped", "reason": reason}

    from rsm_pipeline.explain.unwrap import _unwrap_to_base_model

    base = _unwrap_to_base_model(model)
    cls = type(base).__name__
    save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if cls in {
            "LogisticRegression",
            "RandomForestClassifier",
            "ExtraTreesClassifier",
        }:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType

            initial_types = [("input", FloatTensorType([None, len(feats)]))]
            onx = convert_sklearn(base, initial_types=initial_types)
            save_path.write_bytes(onx.SerializeToString())
        elif cls == "XGBClassifier":
            from onnxmltools.convert import convert_xgboost
            from onnxmltools.convert.common.data_types import FloatTensorType as MtFt

            initial_types = [("input", MtFt([None, len(feats)]))]
            onx = convert_xgboost(base, initial_types=initial_types)
            save_path.write_bytes(onx.SerializeToString())
        elif cls == "LGBMClassifier":
            from onnxmltools.convert import convert_lightgbm
            from onnxmltools.convert.common.data_types import FloatTensorType as MtFt

            initial_types = [("input", MtFt([None, len(feats)]))]
            onx = convert_lightgbm(base, initial_types=initial_types)
            save_path.write_bytes(onx.SerializeToString())
        else:
            return {
                "status": "skipped",
                "reason": f"no ONNX converter for {cls}",
            }
    except Exception as exc:  # noqa: BLE001
        _log.warning("ONNX export failed: %s", exc)
        return {"status": "failed", "reason": str(exc)}

    return {
        "status": "ok",
        "path": str(save_path),
        "format": cls,
        "scope": "model_only",
        "note": (
            "ONNX file contains the leaf estimator only; consumers must apply "
            "preprocessing → feature_engineering → feature_selection externally "
            "before inference. Use bundle.joblib for the full pipeline."
        ),
    }
