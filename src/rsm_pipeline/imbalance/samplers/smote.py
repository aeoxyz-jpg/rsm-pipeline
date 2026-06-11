"""SMOTE / SMOTENC / ADASYN factories."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE, SMOTENC

from rsm_pipeline.imbalance.schema import AdasynConfig, SmoteConfig


def _resolve_categorical_indices(
    feats: list[str], categorical_features: Iterable[str]
) -> list[int]:
    feats_set = set(feats)
    unknown = [c for c in categorical_features if c not in feats_set]
    if unknown:
        raise ValueError(
            f"SMOTE categorical_features unknown to feature list: {unknown}"
        )
    return [feats.index(c) for c in categorical_features]


def _check_all_numeric(X: pd.DataFrame) -> None:
    non_numeric = X.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(
            "SMOTE requires numeric features (or specify categorical_features for "
            f"SMOTENC). Non-numeric columns found: {non_numeric}"
        )


def _make_smote(cfg: SmoteConfig, X: pd.DataFrame, *, random_state: int):
    feats = list(X.columns)
    if cfg.categorical_features:
        cat_idx = _resolve_categorical_indices(feats, cfg.categorical_features)
        return SMOTENC(
            categorical_features=cat_idx,
            sampling_strategy=cfg.sampling_strategy,
            k_neighbors=cfg.k_neighbors,
            random_state=random_state,
        )
    _check_all_numeric(X)
    return SMOTE(
        sampling_strategy=cfg.sampling_strategy,
        k_neighbors=cfg.k_neighbors,
        random_state=random_state,
    )


def _make_adasyn(cfg: AdasynConfig, X: pd.DataFrame, *, random_state: int):
    _check_all_numeric(X)
    return ADASYN(
        sampling_strategy=cfg.sampling_strategy,
        n_neighbors=cfg.n_neighbors,
        random_state=random_state,
    )
