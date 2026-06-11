"""Random / stratified-random / time splitters -> ``SplitResult``."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from rsm_pipeline.config.schema import RsmConfig

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    method: str
    meta: dict[str, Any]


def _label_dist(df: pd.DataFrame, target: str) -> dict[str, int]:
    return {str(int(k)): int(v) for k, v in df[target].value_counts().items()}


def _build_meta(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    method: str,
    extras: dict[str, Any],
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "method": method,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "label_dist": {
            "train": _label_dist(train, target),
            "val": _label_dist(val, target),
            "test": _label_dist(test, target),
        },
    }
    meta.update(extras)
    return meta


def _two_stage_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    stratify_col: pd.Series | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strat_full = stratify_col if stratify_col is not None else None
    train_full, test = train_test_split(
        df,
        test_size=test_ratio,
        random_state=seed,
        stratify=strat_full,
        shuffle=True,
    )
    strat_remain = (
        train_full[stratify_col.name] if stratify_col is not None else None
    )
    val_share = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(
        train_full,
        test_size=val_share,
        random_state=seed,
        stratify=strat_remain,
        shuffle=True,
    )
    return train, val, test


def _random_split(df: pd.DataFrame, cfg: RsmConfig) -> SplitResult:
    rc = cfg.split.random
    train, val, test = _two_stage_split(
        df, rc.train_ratio, rc.val_ratio, rc.test_ratio, cfg.run.seed, None
    )
    meta = _build_meta(train, val, test, cfg.data.target.column, "random", {})
    return SplitResult(train=train, val=val, test=test, method="random", meta=meta)


def _stratified_random_split(df: pd.DataFrame, cfg: RsmConfig) -> SplitResult:
    sr = cfg.split.stratified_random
    col = sr.stratify_by or cfg.data.target.column
    if col not in df.columns:
        raise ValueError(f"stratify_by column {col!r} not in dataframe")
    train, val, test = _two_stage_split(
        df,
        sr.train_ratio,
        sr.val_ratio,
        sr.test_ratio,
        cfg.run.seed,
        df[col],
    )
    meta = _build_meta(
        train,
        val,
        test,
        cfg.data.target.column,
        "stratified_random",
        {"stratify_by": col},
    )
    return SplitResult(
        train=train, val=val, test=test, method="stratified_random", meta=meta
    )


def _time_split(df: pd.DataFrame, cfg: RsmConfig) -> SplitResult:
    tc = cfg.split.time
    date_col = tc.date_column or cfg.data.date_column
    if not date_col or date_col not in df.columns:
        raise ValueError(f"time split date column {date_col!r} not in dataframe")

    df_sorted = df.sort_values(date_col, kind="mergesort").reset_index(drop=True)

    if tc.mode == "ratio":
        n = len(df_sorted)
        n_train = int(n * tc.ratio.train_ratio)
        n_val = int(n * tc.ratio.val_ratio)
        train = df_sorted.iloc[:n_train]
        val = df_sorted.iloc[n_train : n_train + n_val]
        test = df_sorted.iloc[n_train + n_val :]
    else:
        train_end = pd.Timestamp(tc.cutoff.train_end)
        val_end = pd.Timestamp(tc.cutoff.val_end)
        d = df_sorted[date_col]
        train = df_sorted[d <= train_end]
        val = df_sorted[(d > train_end) & (d <= val_end)]
        test = df_sorted[d > val_end]

    for name, fold in (("train", train), ("val", val), ("test", test)):
        if len(fold) == 0:
            raise ValueError(f"time split produced empty fold: {name!r}")

    def _range(fold: pd.DataFrame) -> list[str]:
        return [str(fold[date_col].min()), str(fold[date_col].max())]

    extras = {
        "date_column": date_col,
        "mode": tc.mode,
        "date_range": {
            "train": _range(train),
            "val": _range(val),
            "test": _range(test),
        },
    }
    meta = _build_meta(train, val, test, cfg.data.target.column, "time", extras)
    return SplitResult(train=train, val=val, test=test, method="time", meta=meta)


def split_dataset(df: pd.DataFrame, cfg: RsmConfig) -> SplitResult:
    """Dispatch on ``cfg.split.method``.

    Parameters
    ----------
    df:
        Full dataset as a DataFrame.
    cfg:
        Validated pipeline configuration.

    Returns
    -------
    SplitResult
        Frozen dataclass with train/val/test folds and metadata.

    Examples
    --------
    >>> import pandas as pd
    >>> from rsm_pipeline.config.schema import RsmConfig
    >>> df = pd.DataFrame({"f": range(100), "label": [0, 1] * 50})
    >>> cfg = RsmConfig.model_validate({
    ...     "run": {"name": "demo"},
    ...     "data": {"source": {"format": "csv", "path": "x.csv"}, "target": {"column": "label"}},
    ...     "split": {"method": "random", "random": {"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15}},
    ...     "model": {"kind": "dummy"},
    ...     "evaluation": {"metrics": ["roc_auc"]},
    ... })
    >>> sp = split_dataset(df, cfg)
    >>> sp.method
    'random'
    """
    method = cfg.split.method
    if method == "random":
        result = _random_split(df, cfg)
    elif method == "stratified_random":
        result = _stratified_random_split(df, cfg)
    elif method == "time":
        result = _time_split(df, cfg)
    else:
        raise ValueError(f"unknown split method: {method!r}")

    _log.info(
        "split (%s): train=%d val=%d test=%d",
        result.method,
        result.meta["n_train"],
        result.meta["n_val"],
        result.meta["n_test"],
    )
    return result
