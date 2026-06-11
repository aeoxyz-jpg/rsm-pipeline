"""Partial Dependence Plots — one PNG per feature."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import PartialDependenceDisplay  # noqa: E402

_log = logging.getLogger(__name__)


def _generate_pdp_plots(
    model: Any,
    X_sample: pd.DataFrame,
    top_features: list[str],
    save_dir: Path,
    grid_resolution: int,
    run_dir: Path,
) -> list[str]:
    save_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for feat in top_features:
        try:
            fig, ax = plt.subplots(figsize=(7, 5))
            PartialDependenceDisplay.from_estimator(
                model,
                X_sample,
                [feat],
                grid_resolution=grid_resolution,
                ax=ax,
            )
            fig.tight_layout()
            out = save_dir / f"pdp_{feat}.png"
            fig.savefig(out, dpi=120)
            plt.close(fig)
            paths.append(str(out.relative_to(run_dir)))
        except Exception as exc:  # noqa: BLE001
            _log.warning("PDP failed for feature=%r: %s", feat, exc)
    return paths
