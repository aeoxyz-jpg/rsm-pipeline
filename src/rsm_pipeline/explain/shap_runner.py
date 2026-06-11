"""SHAP explainer selection, run, and plot saving."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rsm_pipeline.explain.unwrap import _unwrap_to_base_model  # noqa: E402

_log = logging.getLogger(__name__)

_TREE_TYPES = {
    "RandomForestClassifier",
    "ExtraTreesClassifier",
    "XGBClassifier",
    "LGBMClassifier",
    "CatBoostClassifier",
}
_LINEAR_TYPES = {"LogisticRegression"}


def _select_explainer(
    model: Any, X_background: pd.DataFrame
) -> tuple[Optional[Any], str]:
    """Return (explainer, kind) where kind in {'tree','linear', or fallback reason}."""
    base = _unwrap_to_base_model(model)
    cls = type(base).__name__
    import shap

    if cls in _TREE_TYPES:
        try:
            return shap.TreeExplainer(base), "tree"
        except Exception as exc:  # noqa: BLE001
            return None, f"tree explainer failed: {exc}"
    if cls in _LINEAR_TYPES:
        try:
            return shap.LinearExplainer(base, X_background), "linear"
        except Exception as exc:  # noqa: BLE001
            return None, f"linear explainer failed: {exc}"
    return None, f"no SHAP explainer for {cls}"


def _normalize_shap_values(raw: Any) -> np.ndarray:
    """Reduce different SHAP return shapes to (n, p) class-1 contributions."""
    if isinstance(raw, list):
        if len(raw) == 2:
            return np.asarray(raw[1])
        return np.asarray(raw[0])
    arr = np.asarray(raw)
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr


def _save_summary_plots(
    shap_vals: np.ndarray,
    X: pd.DataFrame,
    out_dir: Path,
    top_k: int,
    run_dir: Path,
) -> list[str]:
    import shap

    paths: list[str] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for plot_type, fname in (
        ("bar", "shap_summary_bar.png"),
        ("dot", "shap_summary_beeswarm.png"),
    ):
        shap.summary_plot(
            shap_vals, X, plot_type=plot_type, max_display=top_k, show=False
        )
        fig = plt.gcf()
        fig.tight_layout()
        out = out_dir / fname
        fig.savefig(out, dpi=120)
        plt.close(fig)
        paths.append(str(out.relative_to(run_dir)))
    return paths


def _save_local_waterfalls(
    shap_vals: np.ndarray,
    X: pd.DataFrame,
    expected_value: float,
    feature_names: list[str],
    risk_scores: np.ndarray,
    n_local: int,
    out_dir: Path,
    run_dir: Path,
) -> list[str]:
    import shap

    if n_local <= 0 or len(X) == 0:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    top_idx = np.argsort(-risk_scores)[:n_local]
    paths: list[str] = []
    for rank, i in enumerate(top_idx, start=1):
        try:
            expl = shap.Explanation(
                values=shap_vals[i],
                base_values=float(expected_value),
                data=X.iloc[i].values,
                feature_names=feature_names,
            )
            shap.plots.waterfall(expl, show=False)
        except Exception as exc:  # noqa: BLE001
            _log.warning("waterfall failed for sample %d: %s", i, exc)
            plt.close()
            continue
        fig = plt.gcf()
        fig.tight_layout()
        out = out_dir / f"shap_local_top{rank}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        paths.append(str(out.relative_to(run_dir)))
    return paths


def _expected_value_for_class1(explainer: Any) -> float:
    """Best-effort extraction of base value for class 1."""
    ev = getattr(explainer, "expected_value", 0.0)
    if isinstance(ev, (list, tuple, np.ndarray)):
        if len(ev) >= 2:
            return float(ev[1])
        if len(ev) == 1:
            return float(ev[0])
        return 0.0
    return float(ev)
