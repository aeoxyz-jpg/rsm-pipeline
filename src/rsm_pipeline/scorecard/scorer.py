"""Scorer — unified predict_score / predict_proba interface."""

from __future__ import annotations

import dataclasses
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

from rsm_pipeline.scorecard.rank import _score_via_rank


def _score_via_pdo_table(
    X: pd.DataFrame,
    table: pd.DataFrame,
    meta: dict,
) -> np.ndarray:
    """Sum per-feature points via (feature, woe_value) lookup.

    X must contain the same WoE-encoded columns the model was trained on.
    `table` columns include: feature, woe (float), points (int).
    """
    points_lookup: dict[tuple[str, float], int] = {
        (row["feature"], float(row["woe"])): int(row["points"])
        for _, row in table.iterrows()
    }
    # Per-feature LR weight (used to score WoE values not present in the table,
    # e.g. the neutral 0.0 emitted by the encoder for bins/categories unseen at
    # fit time). Points are linear in WoE, so any value can be reconstructed.
    weight_lookup: dict[str, float] = {
        row["feature"]: float(row["weight"]) for _, row in table.iterrows()
    }
    factor = float(meta["factor"])
    base_per_feature = (
        float(meta["offset"]) - float(meta["intercept"]) * factor
    ) / int(meta["n_features"])

    n = len(X)
    total = np.zeros(n, dtype=np.int64)
    for col in meta["features"]:
        if col not in X.columns:
            raise ValueError(
                f"scorer requires column {col!r}; missing from X "
                f"(have {list(X.columns)})"
            )
        vals = X[col].astype(float).to_numpy()
        # Vectorized lookup with float-key matching via rounding
        for w in np.unique(vals):
            key = (col, float(w))
            pts = points_lookup.get(key)
            if pts is None:
                # Try nearest match (float precision)
                candidates = [
                    (k, v)
                    for k, v in points_lookup.items()
                    if k[0] == col and abs(k[1] - float(w)) < 1e-9
                ]
                if candidates:
                    pts = candidates[0][1]
                else:
                    # WoE value not in the table (unseen bin -> neutral WoE).
                    # Reconstruct points from the PDO linear formula.
                    pts = int(
                        round(
                            -weight_lookup.get(col, 0.0) * float(w) * factor
                            + base_per_feature
                        )
                    )
            mask = vals == w
            total[mask] += pts
    return total


@dataclasses.dataclass
class Scorer:
    mode: Literal["pdo", "rank"]
    model: Any
    pdo_table: Optional[pd.DataFrame] = None
    pdo_meta: Optional[dict] = None
    rank_state: Optional[dict] = None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        if self.mode == "pdo":
            assert self.pdo_table is not None and self.pdo_meta is not None
            return _score_via_pdo_table(X, self.pdo_table, self.pdo_meta)
        assert self.rank_state is not None
        probs = self.model.predict_proba(X)[:, 1]
        return _score_via_rank(probs, self.rank_state)
