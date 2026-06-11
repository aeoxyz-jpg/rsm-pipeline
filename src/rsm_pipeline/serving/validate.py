"""Input column validation against TrainedBundle.raw_input_columns."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

_log = logging.getLogger(__name__)


def _validate_input_columns(df: pd.DataFrame, bundle: Any) -> tuple[pd.DataFrame, dict]:
    required = list(bundle.raw_input_columns)
    have = set(df.columns)
    missing = [c for c in required if c not in have]
    if missing:
        raise ValueError(
            f"input is missing required columns: {missing} "
            f"(bundle expects {required})"
        )
    extra = [c for c in df.columns if c not in required]
    if extra:
        _log.warning("dropping extra columns not seen at training: %s", extra)
    df_clean = df[required].copy()
    return df_clean, {"extra_dropped": extra, "n_input_cols": len(required)}
