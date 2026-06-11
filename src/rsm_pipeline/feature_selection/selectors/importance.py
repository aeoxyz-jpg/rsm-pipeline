"""Random-forest feature importance selector."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from rsm_pipeline.feature_selection.selectors._base import _SelectorBase


class ModelImportanceSelector(_SelectorBase):
    def __init__(
        self,
        top_k: Optional[int] = None,
        top_k_pct: Optional[float] = None,
        n_estimators: int = 200,
        max_depth: Optional[int] = 8,
        random_state: Optional[int] = None,
    ):
        self.top_k = top_k
        self.top_k_pct = top_k_pct
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def _fit_kept(self, X: pd.DataFrame, y: Any) -> list[str]:
        if y is None:
            raise ValueError("ModelImportanceSelector.fit requires y")
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=1,
        )
        rf.fit(X.to_numpy(), pd.Series(y).to_numpy())
        importances = dict(zip(X.columns, rf.feature_importances_.tolist()))
        self.importances_ = importances

        n_cols = X.shape[1]
        if self.top_k is not None:
            k = min(self.top_k, n_cols)
        else:
            k = max(1, int(round(n_cols * float(self.top_k_pct))))
        sorted_cols = sorted(X.columns, key=lambda c: -importances[c])
        return list(sorted_cols[:k])

    def drop_reasons(self) -> dict[str, str]:
        dropped = set(self.dropped_columns_)
        return {
            col: f"importance={imp:.6f} not in top-{len(self.kept_columns_)}"
            for col, imp in self.importances_.items()
            if col in dropped
        }
