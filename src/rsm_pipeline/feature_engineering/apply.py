"""Glue: build -> fit on train (with y) -> transform val/test -> return new SplitResult + meta."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer

from rsm_pipeline.config.schema import RsmConfig
from rsm_pipeline.data.splitter import SplitResult
from rsm_pipeline.feature_engineering.factory import (
    _summarize_feature_engineer,
    build_feature_engineer,
)
from rsm_pipeline.preprocessing.dtypes import infer_column_types


def apply_feature_engineering(
    sp: SplitResult,
    cfg: RsmConfig,
    feats: list[str],
    target_col: str,
) -> tuple[SplitResult, ColumnTransformer, dict[str, Any]]:
    """Fit on ``sp.train[feats]`` (with ``sp.train[target_col]`` as ``y``); transform val/test."""
    # Ensure plain Python strings — column names from parquet round-trips may be numpy.str_,
    # which confuses sklearn's ColumnTransformer column lookup.
    feats = [str(c) for c in feats]

    def _coerce_cols(df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with all column names cast to plain Python str."""
        df = df.copy()
        df.columns = [str(c) for c in df.columns]
        return df

    sp = SplitResult(
        train=_coerce_cols(sp.train),
        val=_coerce_cols(sp.val),
        test=_coerce_cols(sp.test),
        method=sp.method,
        meta=sp.meta,
    )

    fe = build_feature_engineer(cfg, sp.train[feats])

    train_arr = fe.fit_transform(sp.train[feats], sp.train[target_col])
    val_arr = fe.transform(sp.val[feats])
    test_arr = fe.transform(sp.test[feats])

    out_cols = list(fe.get_feature_names_out())

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
    fe_meta = _summarize_feature_engineer(fe, types, cfg.feature_engineering)
    return new_sp, fe, fe_meta
