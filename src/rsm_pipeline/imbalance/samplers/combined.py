"""SMOTEENN / SMOTETomek factories."""

from __future__ import annotations

import pandas as pd
from imblearn.combine import SMOTEENN, SMOTETomek

from rsm_pipeline.imbalance.samplers.smote import _check_all_numeric
from rsm_pipeline.imbalance.schema import SmoteennConfig, SmotetomekConfig


def _make_smoteenn(cfg: SmoteennConfig, X: pd.DataFrame, *, random_state: int):
    _check_all_numeric(X)
    return SMOTEENN(
        sampling_strategy=cfg.sampling_strategy,
        random_state=random_state,
    )


def _make_smotetomek(cfg: SmotetomekConfig, X: pd.DataFrame, *, random_state: int):
    _check_all_numeric(X)
    return SMOTETomek(
        sampling_strategy=cfg.sampling_strategy,
        random_state=random_state,
    )
