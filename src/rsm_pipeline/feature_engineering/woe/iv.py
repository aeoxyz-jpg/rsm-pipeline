"""WoE bin / table dataclasses plus the ``compute_iv`` accessor."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WoEBin:
    label: str
    n_total: int
    n_pos: int
    n_neg: int
    pos_rate: float
    woe: float
    iv_contrib: float


@dataclass(frozen=True)
class WoETable:
    column: str
    kind: str
    bins: list[WoEBin]
    iv: float
    n_pos_total: int
    n_neg_total: int
    nan_woe: float


def compute_iv(table: WoETable) -> float:
    """Return ``table.iv``. Convenience accessor for downstream code."""
    return float(table.iv)
