"""Glue: prefit calibrator on val fold, return wrapped model + meta."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from sklearn.calibration import (  # noqa: E402
    CalibratedClassifierCV,
    calibration_curve,
)
from sklearn.frozen import FrozenEstimator  # noqa: E402
from sklearn.metrics import brier_score_loss  # noqa: E402

from rsm_pipeline.config.schema import RsmConfig  # noqa: E402
from rsm_pipeline.data.splitter import SplitResult  # noqa: E402

_log = logging.getLogger(__name__)


_METHOD_MAP = {"platt": "sigmoid", "isotonic": "isotonic"}


def _reliability_records(y_true, y_score, n_bins: int = 10) -> list[dict[str, float]]:
    frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=n_bins)
    return [
        {"bin": int(i), "mean_pred": float(mp), "frac_pos": float(fp)}
        for i, (mp, fp) in enumerate(zip(mean_pred, frac_pos))
    ]


def _plot_compare(y_true, y_pre, y_post, save_path: Path, kind: str) -> None:
    fp_pre, mp_pre = calibration_curve(y_true, y_pre, n_bins=10)
    fp_post, mp_post = calibration_curve(y_true, y_post, n_bins=10)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(mp_pre, fp_pre, "o-", label="pre")
    ax.plot(mp_post, fp_post, "s-", label=f"post ({kind})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    ax.set_xlabel("Mean predicted probability (val)")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration compare — {kind}")
    ax.legend()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def apply_calibration(
    model: Any,
    sp: SplitResult,
    cfg: RsmConfig,
    feats: list[str],
    target: str,
    run_dir: Path,
) -> tuple[Any, dict[str, Any]]:
    """Wrap model with prefit calibrator on val. Returns (new_model, meta).

    Parameters
    ----------
    model:
        Fitted estimator with a ``predict_proba`` method.
    sp:
        Split result; the ``val`` fold is used to fit the calibrator.
    cfg:
        Validated pipeline configuration (must have ``cfg.calibration`` set).
    feats:
        Feature column names to pass to the model.
    target:
        Name of the target column in each fold DataFrame.
    run_dir:
        Root directory for this run; calibration outputs go under ``run_dir/``.

    Returns
    -------
    tuple[Any, dict[str, Any]]
        ``(calibrated_model, meta_dict)``.  If ``cfg.calibration.kind == "none"``
        the original model is returned unchanged.

    Examples
    --------
    >>> import pandas as pd, numpy as np, tempfile, pathlib
    >>> from sklearn.linear_model import LogisticRegression
    >>> from rsm_pipeline.config.schema import RsmConfig
    >>> from rsm_pipeline.data.splitter import SplitResult
    >>> from rsm_pipeline.calibration.apply import apply_calibration
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"f": rng.normal(size=300), "y": (rng.normal(size=300) > 0).astype(int)})
    >>> sp = SplitResult(train=df.iloc[:150], val=df.iloc[150:225], test=df.iloc[225:], method="random", meta={})
    >>> m = LogisticRegression().fit(df.iloc[:150][["f"]], df.iloc[:150]["y"])
    >>> cfg = RsmConfig.model_validate({
    ...     "run": {"name": "t", "seed": 1},
    ...     "data": {"source": {"format": "csv", "path": "x"}, "target": {"column": "y"}},
    ...     "split": {"method": "random", "random": {"train_ratio": 0.5, "val_ratio": 0.25, "test_ratio": 0.25}},
    ...     "model": {"estimator": {"kind": "logreg"}},
    ...     "evaluation": {"thresholds": [0.5]},
    ...     "calibration": {"kind": "isotonic"},
    ... })
    >>> with tempfile.TemporaryDirectory() as td:
    ...     cal, meta = apply_calibration(m, sp, cfg, ["f"], "y", pathlib.Path(td))
    ...     type(cal).__name__
    'CalibratedClassifierCV'
    """
    assert cfg.calibration is not None
    kind = cfg.calibration.kind
    if kind == "none":
        return model, {"kind": "none"}

    method = _METHOD_MAP[kind]
    val_X = sp.val[feats]
    val_y = sp.val[target]

    y_pre = model.predict_proba(val_X)[:, 1]
    if len(set(y_pre.round(6))) < 2:
        raise ValueError(
            "calibration cannot fit on (nearly) constant predictions; "
            "check model quality first"
        )

    cal = CalibratedClassifierCV(FrozenEstimator(model), method=method)
    cal.fit(val_X, val_y)
    y_post = cal.predict_proba(val_X)[:, 1]

    brier_pre = float(brier_score_loss(val_y, y_pre))
    brier_post = float(brier_score_loss(val_y, y_post))

    summary_path = run_dir / cfg.calibration.report.summary_path
    plot_path = run_dir / cfg.calibration.report.plot_path
    summary = {
        "kind": kind,
        "fitted_on": "val",
        "brier_pre": brier_pre,
        "brier_post": brier_post,
        "reliability_pre": _reliability_records(val_y, y_pre),
        "reliability_post": _reliability_records(val_y, y_post),
        "note": (
            "post-calibration metrics on val are biased (calibrator saw this "
            "data); honest post-calibration metrics should be read on test."
        ),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _plot_compare(val_y, y_pre, y_post, plot_path, kind)

    meta = {
        "kind": kind,
        "brier_pre": brier_pre,
        "brier_post": brier_post,
        "summary_path": cfg.calibration.report.summary_path,
        "plot_path": cfg.calibration.report.plot_path,
    }
    return cal, meta
