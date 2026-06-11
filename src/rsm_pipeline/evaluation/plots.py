"""Evaluation plots — Agg backend, PNG output."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

_log = logging.getLogger(__name__)

_FIGSIZE = (7, 5)
_DPI = 120


def _save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)


def _plot_roc(y_true, y_score, save_path: Path, title: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot(fpr, tpr, label="ROC")
    ax.plot([0, 1], [0, 1], "--", color="grey", alpha=0.6)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title)
    ax.legend()
    _save(fig, save_path)


def _plot_pr(y_true, y_score, save_path: Path, title: str) -> None:
    p, r, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot(r, p)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    _save(fig, save_path)


def _plot_ks(y_true, y_score, save_path: Path, title: str) -> None:
    fpr, tpr, thr = roc_curve(y_true, y_score)
    ks_arr = tpr - fpr
    ks_idx = int(np.argmax(np.abs(ks_arr)))
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot(thr, tpr, label="TPR")
    ax.plot(thr, fpr, label="FPR")
    ax.axvline(
        thr[ks_idx],
        color="red",
        linestyle="--",
        label=f"KS={ks_arr[ks_idx]:.3f} @ thr={thr[ks_idx]:.3f}",
    )
    ax.set_xlabel("threshold")
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.legend()
    _save(fig, save_path)


def _plot_confusion(y_true, y_pred_at_threshold, save_path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred_at_threshold, labels=[0, 1])
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred 0", "pred 1"])
    ax.set_yticklabels(["true 0", "true 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    _save(fig, save_path)


def _plot_calibration(y_true, y_score, save_path: Path, title: str) -> None:
    frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=10)
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot(mean_pred, frac_pos, "o-", label="model")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend()
    _save(fig, save_path)


def _plot_score_dist(y_true, y_score, save_path: Path, title: str) -> None:
    y_true_arr = np.asarray(y_true).astype(int)
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.hist(
        np.asarray(y_score)[y_true_arr == 0],
        bins=30,
        alpha=0.5,
        label="class 0",
        density=True,
    )
    ax.hist(
        np.asarray(y_score)[y_true_arr == 1],
        bins=30,
        alpha=0.5,
        label="class 1",
        density=True,
    )
    ax.set_xlabel("score")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend()
    _save(fig, save_path)


def _plot_gain(decile: pd.DataFrame, save_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot(decile["cum_pop_pct"], decile["cum_gain"], label="model")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="random")
    ax.set_xlabel("cum population %")
    ax.set_ylabel("cum gain")
    ax.set_title(title)
    ax.legend()
    _save(fig, save_path)


def _plot_lift(decile: pd.DataFrame, save_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot(decile["cum_pop_pct"], decile["cum_lift"])
    ax.axhline(1.0, color="grey", linestyle="--")
    ax.set_xlabel("cum population %")
    ax.set_ylabel("cumulative lift")
    ax.set_title(title)
    _save(fig, save_path)


def _generate_all_plots(
    y_true,
    y_score,
    decile: pd.DataFrame,
    fold_name: str,
    run_dir: Path,
    thresholds: list[float],
) -> None:
    """Generate the full plot set for a fold, swallowing IO errors."""
    plots_dir = run_dir / "reports" / "plots"
    try:
        _plot_roc(
            y_true, y_score, plots_dir / f"roc_{fold_name}.png", f"ROC ({fold_name})"
        )
        _plot_pr(
            y_true, y_score, plots_dir / f"pr_{fold_name}.png", f"PR ({fold_name})"
        )
        _plot_ks(
            y_true, y_score, plots_dir / f"ks_{fold_name}.png", f"KS ({fold_name})"
        )
        _plot_calibration(
            y_true,
            y_score,
            plots_dir / f"calibration_{fold_name}.png",
            f"Calibration ({fold_name})",
        )
        _plot_score_dist(
            y_true,
            y_score,
            plots_dir / f"score_dist_{fold_name}.png",
            f"Score distribution ({fold_name})",
        )
        _plot_gain(
            decile,
            plots_dir / f"gain_{fold_name}.png",
            f"Cumulative gain ({fold_name})",
        )
        _plot_lift(
            decile,
            plots_dir / f"lift_{fold_name}.png",
            f"Cumulative lift ({fold_name})",
        )
        for t in thresholds:
            y_pred = (np.asarray(y_score) >= t).astype(int)
            _plot_confusion(
                y_true,
                y_pred,
                plots_dir / f"confusion_{fold_name}_t{t:.2f}.png",
                f"Confusion @ thr={t:.2f} ({fold_name})",
            )
    except Exception as exc:  # noqa: BLE001
        _log.warning("plot generation failed: %s", exc)
