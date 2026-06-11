"""Scorecard config — PDO + rank-score parameters."""

from __future__ import annotations

from pydantic import Field, model_validator

from rsm_pipeline._frozen import _Frozen


class ScorecardReportConfig(_Frozen):
    csv_path: str = "reports/scorecard.csv"
    summary_path: str = "reports/scorecard_summary.json"
    artifact_path: str = "artifacts/scorer.joblib"


class ScorecardConfig(_Frozen):
    base_score: int = 600
    base_odds: float = 50.0
    pdo: int = 20
    rank_band_low: int = 300
    rank_band_high: int = 850
    report: ScorecardReportConfig = Field(default_factory=ScorecardReportConfig)

    @model_validator(mode="after")
    def _ranges(self) -> "ScorecardConfig":
        if self.pdo <= 0:
            raise ValueError("scorecard.pdo must be positive")
        if self.base_odds <= 0:
            raise ValueError("scorecard.base_odds must be positive")
        if not (self.rank_band_low < self.rank_band_high):
            raise ValueError("scorecard.rank_band_low must be < rank_band_high")
        return self
