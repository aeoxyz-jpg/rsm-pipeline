"""CSV / Parquet loader with target normalization and date parsing."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from rsm_pipeline.config.schema import RsmConfig

_log = logging.getLogger(__name__)


def _read(cfg: RsmConfig) -> pd.DataFrame:
    src = cfg.data.source
    p = Path(src.path)
    if src.format == "csv":
        return pd.read_csv(
            p,
            sep=src.csv_options.sep,
            encoding=src.csv_options.encoding,
            low_memory=False,
        )
    if src.format == "parquet":
        return pd.read_parquet(p, engine="pyarrow")
    raise ValueError(f"unsupported data.source.format: {src.format!r}")


def _normalize_target(df: pd.DataFrame, cfg: RsmConfig) -> pd.DataFrame:
    col = cfg.data.target.column
    if col not in df.columns:
        raise KeyError(f"target column {col!r} not in dataframe")
    pos = cfg.data.target.positive_class
    raw_unique = pd.unique(df[col].dropna())
    if len(raw_unique) != 2:
        raise ValueError(
            f"target column {col!r} must have exactly two distinct values "
            f"for binary classification (got {sorted(map(repr, raw_unique))})"
        )
    if pos not in raw_unique:
        raise ValueError(
            f"positive_class {pos!r} not present in target column "
            f"(values: {sorted(map(repr, raw_unique))})"
        )
    mapped = (df[col] == pos).astype("int8")
    if mapped.nunique() < 2:
        raise ValueError(
            f"after mapping, target column has only one class "
            f"(positive_class={pos!r})"
        )
    df = df.copy()
    df[col] = mapped
    return df


def _parse_date(df: pd.DataFrame, cfg: RsmConfig) -> pd.DataFrame:
    candidates = []
    if cfg.data.date_column:
        candidates.append(cfg.data.date_column)
    if cfg.split.time and cfg.split.time.date_column:
        candidates.append(cfg.split.time.date_column)
    seen: set[str] = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if c not in df.columns:
            raise KeyError(f"date column {c!r} not in dataframe")
        df = df.copy()
        df[c] = pd.to_datetime(df[c], errors="raise", utc=False)
    return df


def load_dataset(cfg: RsmConfig) -> pd.DataFrame:
    """Read, normalize target, parse date column. Returns the full DataFrame.

    Parameters
    ----------
    cfg : RsmConfig
        Validated pipeline configuration.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe with target mapped to int8 (0/1) and any date
        columns cast to ``datetime64``.

    Examples
    --------
    >>> import tempfile, pathlib
    >>> import pandas as pd
    >>> from rsm_pipeline.config.schema import RsmConfig
    >>> from rsm_pipeline.data.loader import load_dataset
    >>> with tempfile.TemporaryDirectory() as d:
    ...     p = pathlib.Path(d) / "x.csv"
    ...     pd.DataFrame({"f1": [1, 2], "label": [0, 1]}).to_csv(p, index=False)
    ...     cfg = RsmConfig.model_validate({
    ...         "run": {"name": "demo"},
    ...         "data": {"source": {"format": "csv", "path": str(p)}, "target": {"column": "label"}},
    ...         "split": {"method": "stratified_random", "stratified_random": {"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15}},
    ...         "model": {"kind": "dummy"},
    ...         "evaluation": {"metrics": ["roc_auc"]},
    ...     })
    ...     df = load_dataset(cfg)
    ...     df["label"].tolist()
    [0, 1]
    """
    df = _read(cfg)
    df = _normalize_target(df, cfg)
    df = _parse_date(df, cfg)

    target = cfg.data.target.column
    counts = df[target].value_counts().to_dict()
    rate = float(df[target].mean())
    _log.info(
        "loaded %d rows x %d cols from %s (target=%s dist=%s rate=%.4f)",
        len(df),
        df.shape[1],
        cfg.data.source.path,
        target,
        counts,
        rate,
    )
    if cfg.data.date_column or (cfg.split.time and cfg.split.time.date_column):
        date_col = cfg.data.date_column or cfg.split.time.date_column
        _log.info(
            "date column %s range = [%s, %s]",
            date_col,
            df[date_col].min(),
            df[date_col].max(),
        )
    return df
