"""Voting / Stacking config models."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, model_validator

from rsm_pipeline._frozen import _Frozen
from rsm_pipeline.models.schema import NonEnsembleKindConfig


class VotingConfig(_Frozen):
    kind: Literal["voting"] = "voting"
    members: list[NonEnsembleKindConfig] = Field(min_length=2)
    voting: Literal["soft", "hard"] = "soft"
    weights: Optional[list[float]] = None
    n_jobs: int = -1

    @model_validator(mode="after")
    def _weights_match(self) -> "VotingConfig":
        if self.weights is not None and len(self.weights) != len(self.members):
            raise ValueError(
                f"voting.weights length {len(self.weights)} must match "
                f"members length {len(self.members)}"
            )
        return self


class StackingConfig(_Frozen):
    kind: Literal["stacking"] = "stacking"
    members: list[NonEnsembleKindConfig] = Field(min_length=2)
    final_estimator: NonEnsembleKindConfig
    cv: int = 5
    passthrough: bool = False
    n_jobs: int = -1
