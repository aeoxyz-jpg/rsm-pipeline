"""Glue: build -> fit on train -> transform val/test -> return new SplitResult + meta."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer

from rsm_pipeline.config.schema import RsmConfig
from rsm_pipeline.data.splitter import SplitResult
from rsm_pipeline.preprocessing.dtypes import infer_column_types
from rsm_pipeline.preprocessing.factory import (
    _summarize_preprocessor,
    build_preprocessor,
)


def apply_preprocessing(
    sp: SplitResult,
    cfg: RsmConfig,
    feats: list[str],
) -> tuple[SplitResult, ColumnTransformer, dict[str, Any]]:
    """Fit a preprocessor on ``sp.train[feats]`` and transform val/test.

    Returns
    -------
    new_sp : SplitResult
        New SplitResult whose train/val/test DataFrames replace the
        feature columns with their preprocessed versions; non-feature
        columns (target, date) are preserved unchanged.
    pre : ColumnTransformer
        The fitted preprocessor.
    meta : dict
        JSON-friendly summary suitable for ``run.json``.
    """
    pre = build_preprocessor(cfg, sp.train[feats])

    train_arr = pre.fit_transform(sp.train[feats])
    val_arr = pre.transform(sp.val[feats])
    test_arr = pre.transform(sp.test[feats])

    out_cols = list(pre.get_feature_names_out())

    def _rebuild(orig: pd.DataFrame, arr) -> pd.DataFrame:
        new_feats_df = pd.DataFrame(arr, columns=out_cols, index=orig.index)
        keep = [c for c in orig.columns if c not in feats]
        return pd.concat([orig[keep], new_feats_df], axis=1)

    new_sp = SplitResult(
        train=_rebuild(sp.train, train_arr),
        val=_rebuild(sp.val, val_arr),
        test=_rebuild(sp.test, test_arr),
        method=sp.method,
        meta=sp.meta,
    )

    types = infer_column_types(sp.train[feats])
    pre_meta = _summarize_preprocessor(pre, types, cfg.preprocessing)
    return new_sp, pre, pre_meta
