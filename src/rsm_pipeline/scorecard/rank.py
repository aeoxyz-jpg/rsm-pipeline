"""Rank-score fallback — ECDF on train probabilities -> integer band."""

from __future__ import annotations

import numpy as np


def _build_rank_state(train_probs: np.ndarray, low: int, high: int) -> dict:
    sorted_train = np.sort(np.asarray(train_probs, dtype=float))
    return {
        "sorted_train_probs": sorted_train,
        "low": int(low),
        "high": int(high),
    }


def _score_via_rank(probs: np.ndarray, state: dict) -> np.ndarray:
    """Map probs -> integer score in [low, high]; higher prob -> lower score."""
    probs = np.asarray(probs, dtype=float)
    sorted_train = state["sorted_train_probs"]
    n = max(len(sorted_train), 1)
    ranks = np.searchsorted(sorted_train, probs, side="right") / n
    span = state["high"] - state["low"]
    return np.round(state["low"] + (1.0 - ranks) * span).astype(int)
