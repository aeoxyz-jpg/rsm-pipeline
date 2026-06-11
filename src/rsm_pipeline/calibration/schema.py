"""Calibration config — Platt / Isotonic / none."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from rsm_pipeline._frozen import _Frozen


class CalibrationReportConfig(_Frozen):
    summary_path: str = "reports/calibration_summary.json"
    plot_path: str = "reports/plots/calibration_compare.png"


class CalibrationConfig(_Frozen):
    kind: Literal["none", "platt", "isotonic"] = "none"
    report: CalibrationReportConfig = Field(default_factory=CalibrationReportConfig)
