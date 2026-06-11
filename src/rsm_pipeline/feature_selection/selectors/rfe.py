"""RFE selector with fixed LogisticRegression(L2) estimator."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

from rsm_pipeline.feature_selection.selectors._base import _SelectorBase


class RFESelector(_SelectorBase):
    def __init__(self, n_features_to_select: int, step: int = 1):
        self.n_features_to_select = n_features_to_select
        self.step = step

    def _fit_kept(self, X: pd.DataFrame, y: Any) -> list[str]:
        if y is None:
            raise ValueError("RFESelector.fit requires y")
        n_target = min(self.n_features_to_select, X.shape[1])
        estimator = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=0,
        )
        rfe = RFE(estimator=estimator, n_features_to_select=n_target, step=self.step)
        rfe.fit(X.to_numpy(), pd.Series(y).to_numpy())
        self.ranking_ = dict(zip(X.columns, rfe.ranking_.tolist()))
        return [c for c, m in zip(X.columns, rfe.support_) if m]

    def drop_reasons(self) -> dict[str, str]:
        dropped = set(self.dropped_columns_)
        return {
            col: f"rfe_rank={rank} (kept top {self.n_features_to_select})"
            for col, rank in self.ranking_.items()
            if col in dropped
        }
