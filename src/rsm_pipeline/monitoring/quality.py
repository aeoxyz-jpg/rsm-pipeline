"""Per-feature data quality checks: missing-rate, KS-test, new categorical levels."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def _data_quality_checks(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for col in columns:
        ref_s = reference[col]
        cur_s = current[col]
        info: dict[str, Any] = {
            "missing_rate_ref": float(ref_s.isna().mean()),
            "missing_rate_curr": float(cur_s.isna().mean()),
        }
        info["missing_rate_delta"] = (
            info["missing_rate_curr"] - info["missing_rate_ref"]
        )
        if pd.api.types.is_numeric_dtype(ref_s):
            info["dtype"] = "numeric"
            ref_arr = ref_s.dropna().to_numpy()
            cur_arr = cur_s.dropna().to_numpy()
            if len(ref_arr) > 0 and len(cur_arr) > 0:
                stat = ks_2samp(ref_arr, cur_arr)
                info["ks_p_value"] = float(stat.pvalue)
                info["ks_statistic"] = float(stat.statistic)
            else:
                info["ks_p_value"] = None
                info["ks_statistic"] = None
        else:
            info["dtype"] = "categorical"
            ref_levels = set(ref_s.dropna().astype(str).unique())
            cur_levels = set(cur_s.dropna().astype(str).unique())
            info["new_levels"] = sorted(cur_levels - ref_levels)
        out[col] = info
    return out
