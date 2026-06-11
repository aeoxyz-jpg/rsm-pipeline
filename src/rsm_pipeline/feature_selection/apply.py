"""Glue: build -> fit on train (with y) -> transform val/test -> write CSV + meta."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from rsm_pipeline.config.schema import RsmConfig
from rsm_pipeline.data.splitter import SplitResult
from rsm_pipeline.feature_selection.factory import (
    _summarize_feature_selection,
    _write_report_csv,
    build_feature_selector,
)


def apply_feature_selection(
    sp: SplitResult,
    cfg: RsmConfig,
    feats: list[str],
    target_col: str,
    run_dir: Path,
) -> tuple[SplitResult, Pipeline, dict[str, Any]]:
    """Fit chained selectors on train (with y); transform val/test.

    Writes ``run_dir / cfg.feature_selection.report.csv_path``.
    """
    pipe = build_feature_selector(cfg)
    pipe.fit(sp.train[feats], sp.train[target_col])

    train_kept = pipe.transform(sp.train[feats])
    val_kept = pipe.transform(sp.val[feats])
    test_kept = pipe.transform(sp.test[feats])

    def _rebuild(orig: pd.DataFrame, kept: pd.DataFrame) -> pd.DataFrame:
        keep_meta = [c for c in orig.columns if c not in feats]
        return pd.concat([orig[keep_meta], kept], axis=1)

    new_sp = SplitResult(
        train=_rebuild(sp.train, train_kept),
        val=_rebuild(sp.val, val_kept),
        test=_rebuild(sp.test, test_kept),
        method=sp.method,
        meta=sp.meta,
    )

    csv_rel = cfg.feature_selection.report.csv_path
    csv_full = run_dir / csv_rel
    _write_report_csv(pipe, csv_full)

    fs_meta = _summarize_feature_selection(pipe, feats, csv_rel)
    return new_sp, pipe, fs_meta
