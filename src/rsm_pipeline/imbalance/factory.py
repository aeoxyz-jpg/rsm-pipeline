"""Factory: SamplerConfig -> imblearn sampler instance, plus auto resolver."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from rsm_pipeline.imbalance.samplers._base import (
    _class_counts,
    _compute_class_weight,
    _minority_ratio,
)
from rsm_pipeline.imbalance.samplers.combined import (
    _make_smoteenn,
    _make_smotetomek,
)
from rsm_pipeline.imbalance.samplers.random import (
    _make_random_oversampler,
    _make_random_undersampler,
)
from rsm_pipeline.imbalance.samplers.smote import _make_adasyn, _make_smote
from rsm_pipeline.imbalance.schema import (
    AdasynConfig,
    AutoConfig,
    ClassWeightConfig,
    NoneSamplerConfig,
    RandomOverConfig,
    RandomUnderConfig,
    SamplerConfig,
    SmoteConfig,
    SmoteennConfig,
    SmotetomekConfig,
)


def _resolve_auto(cfg: AutoConfig, y: pd.Series) -> SamplerConfig:
    """Replace AutoConfig with a concrete config based on minority ratio."""
    ratio = _minority_ratio(y)
    if ratio < cfg.high_imbalance_threshold:
        return SmoteConfig()
    if ratio < cfg.moderate_imbalance_threshold:
        return ClassWeightConfig()
    return NoneSamplerConfig()


def build_sampler(
    cfg: SamplerConfig, X: pd.DataFrame, *, random_state: int
) -> Optional[Any]:
    """Return an imblearn sampler, or None for none / class_weight."""
    if isinstance(cfg, (NoneSamplerConfig, ClassWeightConfig)):
        return None
    if isinstance(cfg, RandomOverConfig):
        return _make_random_oversampler(cfg, random_state=random_state)
    if isinstance(cfg, RandomUnderConfig):
        return _make_random_undersampler(cfg, random_state=random_state)
    if isinstance(cfg, SmoteConfig):
        return _make_smote(cfg, X, random_state=random_state)
    if isinstance(cfg, AdasynConfig):
        return _make_adasyn(cfg, X, random_state=random_state)
    if isinstance(cfg, SmoteennConfig):
        return _make_smoteenn(cfg, X, random_state=random_state)
    if isinstance(cfg, SmotetomekConfig):
        return _make_smotetomek(cfg, X, random_state=random_state)
    raise TypeError(f"unsupported sampler config: {type(cfg).__name__}")


def _summarize_imbalance(
    *,
    kind: str,
    auto_resolved_to: Optional[str],
    before: pd.Series,
    after: pd.Series,
    sampling_strategy: Any,
    categorical_features: Optional[list[str]],
    class_weight: Optional[dict[str, float]],
    meta_cols_dropped_on_synthetic: list[str],
) -> dict[str, Any]:
    before_d = _class_counts(before)
    after_d = _class_counts(after)
    out: dict[str, Any] = {
        "kind": kind,
        "auto_resolved_to": auto_resolved_to,
        "before": {
            **before_d,
            "minority_ratio": _minority_ratio(before),
            "n_total": int(before.shape[0]),
        },
        "after": {
            **after_d,
            "minority_ratio": _minority_ratio(after),
            "n_total": int(after.shape[0]),
        },
        "sampling_strategy": sampling_strategy,
        "categorical_features": categorical_features,
        "meta_cols_dropped_on_synthetic": meta_cols_dropped_on_synthetic,
    }
    if class_weight is not None:
        out["class_weight"] = class_weight
    return out


def _write_report_json(meta: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


__all__ = [
    "build_sampler",
    "_resolve_auto",
    "_summarize_imbalance",
    "_write_report_json",
    "_compute_class_weight",
]
