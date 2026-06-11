"""TrainedBundle — joblib-portable artifact wrapping the full pipeline."""

from __future__ import annotations

import dataclasses
import logging
import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


def _build_meta() -> dict[str, str]:
    """Snapshot relevant package versions at bundle creation."""
    import sys

    info: dict[str, str] = {
        "python": ".".join(map(str, sys.version_info[:3])),
    }
    for mod in ("sklearn", "xgboost", "lightgbm"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            info[mod] = "not_installed"
    return info


@dataclasses.dataclass
class TrainedBundle:
    preprocessing: Optional[Any]
    feature_engineering: Optional[Any]
    feature_selection: Optional[Any]
    model: Any
    scorer: Optional[Any]
    feats_after_fs: list[str]
    target: str
    date_column: Optional[str]
    raw_input_columns: list[str]
    build_meta: dict[str, str] = dataclasses.field(default_factory=_build_meta)

    def _transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        # Select only the columns the bundle was trained on; callers may pass
        # the full CSV (including target, date, etc.) without issue.
        available = [c for c in self.raw_input_columns if c in raw.columns]
        df = raw[available].copy()
        if self.preprocessing is not None:
            df = self._apply_step(self.preprocessing, df)
        if self.feature_engineering is not None:
            df = self._apply_step(self.feature_engineering, df)
        if self.feature_selection is not None:
            df = self.feature_selection.transform(df)
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(
                    df,
                    columns=self.feats_after_fs,
                )
        return df

    @staticmethod
    def _apply_step(transformer: Any, df: pd.DataFrame) -> pd.DataFrame:
        out = transformer.transform(df)
        if isinstance(out, pd.DataFrame):
            return out
        # ColumnTransformer with default verbose_feature_names_out=False can
        # return ndarray; recover columns via get_feature_names_out.
        # Cast to plain str to avoid numpy.str_ causing downstream CT failures.
        try:
            cols = [str(c) for c in transformer.get_feature_names_out()]
        except Exception:  # noqa: BLE001
            cols = [f"x{i}" for i in range(out.shape[1])]
        return pd.DataFrame(out, columns=cols, index=df.index)

    def predict_proba(self, raw: pd.DataFrame) -> np.ndarray:
        df = self._transform(raw)
        return self.model.predict_proba(df[self.feats_after_fs])

    def predict(self, raw: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(raw)[:, 1] >= threshold).astype(int)

    def predict_score(self, raw: pd.DataFrame) -> np.ndarray:
        if self.scorer is None:
            raise RuntimeError(
                "this bundle has no scorer; rerun training with cfg.scorecard set"
            )
        df = self._transform(raw)
        return self.scorer.predict_score(df[self.feats_after_fs])

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        # Cross-version warning at load time
        try:
            import sklearn

            current = getattr(sklearn, "__version__", "unknown")
            recorded = self.build_meta.get("sklearn", "unknown")
            if current != recorded:
                warnings.warn(
                    f"TrainedBundle was built with sklearn={recorded}; "
                    f"current is {current}. Predictions may differ.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        except ImportError:
            pass
