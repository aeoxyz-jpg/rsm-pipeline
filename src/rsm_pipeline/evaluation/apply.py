"""Glue: scalar metrics + decile + plots per fold."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rsm_pipeline.config.schema import RsmConfig
from rsm_pipeline.data.splitter import SplitResult
from rsm_pipeline.evaluation.decile import _compute_decile_lift
from rsm_pipeline.evaluation.metrics import _compute_scalar_metrics
from rsm_pipeline.evaluation.plots import _generate_all_plots

_log = logging.getLogger(__name__)


def apply_full_evaluation(
    model: Any,
    sp: SplitResult,
    cfg: RsmConfig,
    feats: list[str],
    target: str,
    run_dir: Path,
) -> dict[str, dict]:
    """Compute panel + decile + plots for val and test. Return {fold: metrics}.

    Parameters
    ----------
    model:
        Fitted estimator with a ``predict_proba`` method.
    sp:
        Split result containing val and test folds.
    cfg:
        Validated pipeline configuration.
    feats:
        Feature column names to pass to the model.
    target:
        Name of the target column in each fold DataFrame.
    run_dir:
        Root directory for this run; reports are written under ``run_dir/reports/``.

    Returns
    -------
    dict[str, dict]
        Mapping ``fold_name -> scalar_metrics_dict`` for ``"val"`` and ``"test"``.

    Examples
    --------
    >>> import pandas as pd, numpy as np, pathlib, tempfile
    >>> from sklearn.linear_model import LogisticRegression
    >>> from rsm_pipeline.config.schema import RsmConfig
    >>> from rsm_pipeline.data.splitter import SplitResult
    >>> from rsm_pipeline.evaluation.apply import apply_full_evaluation
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"f": rng.normal(size=200), "y": rng.integers(0, 2, 200)})
    >>> sp = SplitResult(train=df.iloc[:100], val=df.iloc[100:150],
    ...                  test=df.iloc[150:], method="random", meta={})
    >>> m = LogisticRegression().fit(df.iloc[:100][["f"]], df.iloc[:100]["y"])
    >>> cfg = RsmConfig.model_validate({
    ...     "run": {"name": "t", "seed": 1},
    ...     "data": {"source": {"format": "csv", "path": "x"}, "target": {"column": "y"}},
    ...     "split": {"method": "random", "random": {"train_ratio": 0.5, "val_ratio": 0.25, "test_ratio": 0.25}},
    ...     "model": {"estimator": {"kind": "logreg"}},
    ...     "evaluation": {"thresholds": [0.5]},
    ... })
    >>> with tempfile.TemporaryDirectory() as td:
    ...     out = apply_full_evaluation(m, sp, cfg, ["f"], "y", pathlib.Path(td))
    ...     "roc_auc" in out["val"]
    True
    """
    out: dict[str, dict] = {}
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    for fold_name, fold_df in (("val", sp.val), ("test", sp.test)):
        y_true = fold_df[target]
        proba = model.predict_proba(fold_df[feats])
        if proba.shape[1] >= 2:
            y_score = proba[:, 1]
        else:
            _log.warning(
                "fold=%s predict_proba has 1 column; falling back to 0.5 scores",
                fold_name,
            )
            y_score = np.full(len(fold_df), 0.5)

        metrics = _compute_scalar_metrics(y_true, y_score, cfg.evaluation.thresholds)
        decile = _compute_decile_lift(y_true, y_score)

        (reports_dir / f"metrics_{fold_name}.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        decile.to_csv(
            reports_dir / f"decile_lift_{fold_name}.csv",
            encoding="utf-8",
            index=False,
        )

        if cfg.evaluation.plots.enabled:
            _generate_all_plots(
                y_true,
                y_score,
                decile,
                fold_name,
                run_dir,
                cfg.evaluation.thresholds,
            )

        out[fold_name] = metrics
    return out
