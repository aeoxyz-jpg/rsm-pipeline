"""Export config — joblib (always) + optional ONNX/PMML."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from rsm_pipeline._frozen import _Frozen


class ExportConfig(_Frozen):
    kinds: list[Literal["joblib", "onnx", "pmml"]] = Field(
        default_factory=lambda: ["joblib"]
    )
    bundle_path: str = "artifacts/bundle.joblib"
    onnx_path: str = "artifacts/model.onnx"
    pmml_path: str = "artifacts/model.pmml"
    summary_path: str = "reports/export_summary.json"

    @field_validator("kinds", mode="after")
    @classmethod
    def _has_joblib(cls, v: list[str]) -> list[str]:
        if "joblib" not in v:
            v = ["joblib"] + list(v)
        return v
