"""Glue: build Scorer (PDO or rank), persist artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from rsm_pipeline.config.schema import RsmConfig
from rsm_pipeline.data.splitter import SplitResult
from rsm_pipeline.scorecard.pdo import (
    _build_pdo_table,
    _collect_woe_encoders,
    _is_pdo_eligible,
    _unwrap_to_lr,
)
from rsm_pipeline.scorecard.rank import _build_rank_state
from rsm_pipeline.scorecard.scorer import Scorer

_log = logging.getLogger(__name__)


def apply_scorecard(
    model: Any,
    fe_pipeline: Any,
    sp: SplitResult,
    cfg: RsmConfig,
    feats: list[str],
    target: str,
    run_dir: Path,
) -> tuple[Scorer, dict[str, Any]]:
    """Decide PDO vs rank-score, persist artifacts, return (scorer, meta)."""
    assert cfg.scorecard is not None
    sc_cfg = cfg.scorecard
    reports_dir = run_dir / "reports"
    artifact_path = run_dir / sc_cfg.report.artifact_path
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    eligible = _is_pdo_eligible(model, fe_pipeline)
    fallback_reason: str | None = None
    if not eligible:
        if _unwrap_to_lr(model) is None:
            fallback_reason = (
                f"unwrapped model is not LogisticRegression "
                f"(got {type(model).__name__})"
            )
        else:
            fallback_reason = "feature pipeline does not consist solely of WoE encoders"

    if eligible:
        lr = _unwrap_to_lr(model)
        woe_encoders = _collect_woe_encoders(fe_pipeline)
        feature_order = list(feats)
        missing = [f for f in feature_order if f not in woe_encoders]
        assert (
            not missing
        ), f"PDO eligibility passed but features without WoE encoder: {missing}"
        table, meta = _build_pdo_table(
            lr,
            woe_encoders,
            feature_order,
            sc_cfg.base_score,
            sc_cfg.base_odds,
            sc_cfg.pdo,
        )
        csv_path = run_dir / sc_cfg.report.csv_path
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(csv_path, encoding="utf-8", index=False)
        summary_path = run_dir / sc_cfg.report.summary_path
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "mode": "pdo",
            **{k: v for k, v in meta.items() if k != "features"},
            "n_features": meta["n_features"],
            "csv_path": sc_cfg.report.csv_path,
            "artifact_path": sc_cfg.report.artifact_path,
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        scorer = Scorer(
            mode="pdo",
            model=model,
            pdo_table=table,
            pdo_meta=meta,
        )
        joblib.dump(scorer, artifact_path)
        return scorer, summary

    # Rank-score fallback
    _log.warning("scorecard falling back to rank-score: %s", fallback_reason)
    train_probs = model.predict_proba(sp.train[feats])[:, 1]
    rank_state = _build_rank_state(
        train_probs, sc_cfg.rank_band_low, sc_cfg.rank_band_high
    )
    summary = {
        "mode": "rank",
        "fallback_reason": fallback_reason,
        "n_train": int(len(train_probs)),
        "band": [int(sc_cfg.rank_band_low), int(sc_cfg.rank_band_high)],
        "artifact_path": sc_cfg.report.artifact_path,
    }
    summary_path = run_dir / sc_cfg.report.summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    scorer = Scorer(
        mode="rank",
        model=model,
        rank_state=rank_state,
    )
    joblib.dump(scorer, artifact_path)
    return scorer, summary
