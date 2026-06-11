"""Best-effort PMML export. Requires the optional [pmml] extra."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


def _export_pmml(
    model: Any,
    preprocessing: Any,
    feature_engineering: Any,
    feature_selection: Any,
    save_path: Path,
) -> dict[str, Any]:
    """Try PMML conversion; gracefully skip if sklearn2pmml is missing."""
    try:
        from sklearn2pmml import sklearn2pmml as s2p_export  # noqa: F401
        from sklearn2pmml.pipeline import PMMLPipeline  # noqa: F401
    except ImportError:
        return {
            "status": "skipped",
            "reason": (
                "sklearn2pmml not installed; install with "
                "`pip install '.[pmml]'` (requires Java)"
            ),
        }

    # Detect WoE encoders — sklearn2pmml has no converter for them.
    try:
        from rsm_pipeline.feature_engineering.woe.categorical import (
            CategoricalWoEEncoder,
        )
        from rsm_pipeline.feature_engineering.woe.numerical import (
            NumericalWoEEncoder,
        )
    except ImportError:
        return {"status": "failed", "reason": "WoE module import failed"}

    if feature_engineering is not None:
        for _name, t, _cols in getattr(feature_engineering, "transformers_", []):
            if isinstance(t, (NumericalWoEEncoder, CategoricalWoEEncoder)):
                return {
                    "status": "skipped",
                    "reason": "WoE encoders not supported by sklearn2pmml",
                }

    return {
        "status": "skipped",
        "reason": (
            "PMML export of compound (preprocessing → FE → FS → model) "
            "pipelines requires manual PMMLPipeline construction; not "
            "implemented in #10a"
        ),
    }
