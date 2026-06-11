"""score_batch — apply a TrainedBundle to a DataFrame."""

from __future__ import annotations

from typing import Any

import pandas as pd

from rsm_pipeline.serving.validate import _validate_input_columns


def score_batch(
    bundle: Any,
    input_df: pd.DataFrame,
    *,
    threshold: float = 0.5,
    include_score: bool = False,
) -> pd.DataFrame:
    """Score a DataFrame; return original cols + proba_1, predict (and score)."""
    clean, _meta = _validate_input_columns(input_df, bundle)
    proba = bundle.predict_proba(clean)
    proba_1 = proba[:, 1]
    out = input_df.copy()
    out["proba_1"] = proba_1
    out["predict"] = (proba_1 >= threshold).astype(int)
    if include_score:
        if bundle.scorer is None:
            raise RuntimeError(
                "bundle has no scorer; train with cfg.scorecard to enable "
                "--include-score"
            )
        out["score"] = bundle.predict_score(clean)
    return out
