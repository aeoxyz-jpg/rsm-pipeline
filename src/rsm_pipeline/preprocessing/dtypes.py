"""Column-dtype inference: numeric vs categorical."""

from __future__ import annotations

import pandas as pd


def infer_column_types(df: pd.DataFrame) -> dict[str, str]:
    """Return ``{col: 'numeric' | 'categorical'}`` for every column in ``df``.

    Numeric: any numeric dtype that isn't bool.
    Categorical: object, category, bool. Datetime / timedelta should not
    reach this layer (caller drops them); they are reported as
    ``'categorical'`` defensively.
    """
    out: dict[str, str] = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            out[c] = "numeric"
        elif (
            pd.api.types.is_object_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(s)
        ):
            out[c] = "categorical"
        else:
            out[c] = "categorical"
    return out
