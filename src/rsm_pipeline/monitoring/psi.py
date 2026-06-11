"""PSI on score distribution with reference quantile binning."""

from __future__ import annotations

import numpy as np


_EPS = 1e-6


def _bin_by_quantiles(reference: np.ndarray, n_bins: int) -> np.ndarray:
    """Return bin edges based on reference quantiles, with -inf/+inf guards."""
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(reference, qs))
    if len(edges) < 2:
        edges = np.array([reference.min(), reference.max() + 1e-9])
    edges = np.concatenate([[-np.inf], edges[1:-1], [np.inf]])
    return edges


def _proportions(arr: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(arr, bins=edges)
    total = max(counts.sum(), 1)
    p = counts / total
    return p + _EPS  # smooth to avoid log(0)


def _compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    n_bins: int = 10,
) -> tuple[float, list[dict]]:
    """Return (psi_value, per_bin_breakdown). Both inputs must be 1-D float arrays."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    edges = _bin_by_quantiles(ref, n_bins)
    p_ref = _proportions(ref, edges)
    p_cur = _proportions(cur, edges)
    psi_terms = (p_cur - p_ref) * np.log(p_cur / p_ref)
    psi = float(psi_terms.sum())
    breakdown = [
        {
            "bin": i,
            "edge_low": (None if not np.isfinite(edges[i]) else float(edges[i])),
            "edge_high": (
                None if not np.isfinite(edges[i + 1]) else float(edges[i + 1])
            ),
            "p_ref": float(p_ref[i] - _EPS),
            "p_curr": float(p_cur[i] - _EPS),
            "contribution": float(psi_terms[i]),
        }
        for i in range(len(edges) - 1)
    ]
    return psi, breakdown


def _tier(psi: float, minor: float, major: float) -> str:
    if psi < minor:
        return "stable"
    if psi < major:
        return "minor_shift"
    return "major_shift"
