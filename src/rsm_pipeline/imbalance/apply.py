"""Glue: resample train fold only, write report, return new SplitResult."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from rsm_pipeline.config.schema import RsmConfig
from rsm_pipeline.data.splitter import SplitResult
from rsm_pipeline.imbalance.factory import (
    _compute_class_weight,
    _resolve_auto,
    _summarize_imbalance,
    _write_report_json,
    build_sampler,
)
from rsm_pipeline.imbalance.schema import (
    AutoConfig,
    ClassWeightConfig,
    NoneSamplerConfig,
    SmoteConfig,
)

_log = logging.getLogger(__name__)


def _reconstruct_train(
    original: pd.DataFrame,
    feats: list[str],
    target: str,
    X_res: pd.DataFrame,
    y_res: pd.Series,
) -> tuple[pd.DataFrame, list[str]]:
    """Rebuild a train DataFrame keeping meta columns aligned to sampled rows.

    ``original`` must have a clean 0..N-1 RangeIndex (caller resets it).
    Real rows in X_res have indices in [0, len(original)); synthetic rows
    (SMOTE/ADASYN/combined) have indices >= len(original) and get NaN meta.

    Returns (df, meta_cols).
    """
    meta_cols = [c for c in original.columns if c not in feats and c != target]
    n_orig = len(original)
    sampled_idx = pd.Index(X_res.index)

    feats_part = X_res.reset_index(drop=True)
    target_part = pd.Series(y_res.values, name=target).to_frame()
    parts: list[pd.DataFrame] = [feats_part, target_part]

    if meta_cols:
        meta_full = pd.DataFrame(
            {c: pd.array([pd.NA] * len(sampled_idx)) for c in meta_cols}
        )
        is_real = sampled_idx < n_orig
        real_mask = np.asarray(is_real)
        real_positions = sampled_idx[real_mask]
        meta_real = original.loc[real_positions, meta_cols].reset_index(drop=True)
        for c in meta_cols:
            meta_full.loc[real_mask, c] = meta_real[c].to_numpy()
        parts.insert(0, meta_full)

    df = pd.concat(parts, axis=1)
    return df[meta_cols + feats + [target]], meta_cols


def apply_imbalance(
    sp: SplitResult,
    cfg: RsmConfig,
    feats: list[str],
    target: str,
    run_dir: Path,
) -> tuple[SplitResult, dict[str, Any]]:
    """Resample train only; val/test returned with object identity preserved."""
    assert cfg.imbalance is not None, "apply_imbalance called without cfg.imbalance"
    seed = cfg.run.seed
    sampler_cfg = cfg.imbalance.sampler

    # Resolve auto
    auto_resolved_to: Optional[str] = None
    original_kind = sampler_cfg.kind
    if isinstance(sampler_cfg, AutoConfig):
        sampler_cfg = _resolve_auto(sampler_cfg, sp.train[target])
        auto_resolved_to = sampler_cfg.kind
        _log.info("imbalance auto resolved to kind=%s", auto_resolved_to)

    before_y = sp.train[target]

    # Short-circuit: none / class_weight do not resample
    if isinstance(sampler_cfg, (NoneSamplerConfig, ClassWeightConfig)):
        class_weight = None
        if isinstance(sampler_cfg, ClassWeightConfig):
            class_weight = _compute_class_weight(sampler_cfg.weight, before_y)
        meta = _summarize_imbalance(
            kind=sampler_cfg.kind,
            auto_resolved_to=auto_resolved_to,
            before=before_y,
            after=before_y,
            sampling_strategy=None,
            categorical_features=None,
            class_weight=class_weight,
            meta_cols_dropped_on_synthetic=[],
        )
        if original_kind != sampler_cfg.kind:
            meta["original_kind"] = original_kind
        _write_report_json(meta, run_dir / cfg.imbalance.report.json_path)
        return sp, meta  # identity preserved

    # Resample path — reset index so X_res positions are interpretable as
    # 0..n_orig-1 for real rows, >= n_orig for synthetic rows.
    train_reset = sp.train.reset_index(drop=True)
    X_in = train_reset[feats]
    y_in = train_reset[target]
    sampler = build_sampler(sampler_cfg, X_in, random_state=seed)
    try:
        X_res, y_res = sampler.fit_resample(X_in, y_in)
    except ValueError as exc:
        if "n_neighbors" in str(exc) or "n_samples_fit" in str(exc):
            n_min = int(before_y.value_counts().min())
            req = getattr(sampler_cfg, "k_neighbors", None) or getattr(
                sampler_cfg, "n_neighbors", None
            )
            if req is not None:
                raise ValueError(
                    f"{sampler_cfg.kind} requires at least k_neighbors+1={req + 1} "
                    f"minority samples; got {n_min}. Reduce k_neighbors/n_neighbors "
                    f"or use random_oversample."
                ) from exc
        raise
    if not isinstance(X_res, pd.DataFrame):
        X_res = pd.DataFrame(X_res, columns=feats)
    if not isinstance(y_res, pd.Series):
        y_res = pd.Series(y_res, name=target)

    new_train, meta_cols = _reconstruct_train(train_reset, feats, target, X_res, y_res)
    new_sp = SplitResult(
        train=new_train,
        val=sp.val,
        test=sp.test,
        method=sp.method,
        meta=sp.meta,
    )
    assert new_sp.val is sp.val and new_sp.test is sp.test

    sampling_strategy = getattr(sampler_cfg, "sampling_strategy", None)
    categorical_features = None
    if isinstance(sampler_cfg, SmoteConfig):
        categorical_features = sampler_cfg.categorical_features

    meta = _summarize_imbalance(
        kind=sampler_cfg.kind,
        auto_resolved_to=auto_resolved_to,
        before=before_y,
        after=y_res,
        sampling_strategy=sampling_strategy,
        categorical_features=categorical_features,
        class_weight=None,
        meta_cols_dropped_on_synthetic=(
            meta_cols if len(new_train) > len(sp.train) else []
        ),
    )
    if original_kind != sampler_cfg.kind:
        meta["original_kind"] = original_kind
    _write_report_json(meta, run_dir / cfg.imbalance.report.json_path)
    return new_sp, meta
