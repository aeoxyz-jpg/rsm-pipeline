"""CSV / Parquet read/write helpers for the batch CLI."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _read_input(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8")
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path, engine="pyarrow")
    raise ValueError(f"unsupported input extension: {suffix!r}")


def _write_output(df: pd.DataFrame, path: Path) -> None:
    suffix = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".csv":
        df.to_csv(path, index=False, encoding="utf-8")
    elif suffix in (".parquet", ".pq"):
        df.to_parquet(path, engine="pyarrow", index=False)
    else:
        raise ValueError(f"unsupported output extension: {suffix!r}")
