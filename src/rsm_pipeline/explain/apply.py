"""apply_explain: SHAP + PDP + native importance."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rsm_pipeline.config.schema import RsmConfig
from rsm_pipeline.data.splitter import SplitResult
from rsm_pipeline.explain.pdp import _generate_pdp_plots
from rsm_pipeline.explain.shap_runner import (
    _expected_value_for_class1,
    _normalize_shap_values,
    _save_local_waterfalls,
    _save_summary_plots,
    _select_explainer,
)
from rsm_pipeline.explain.unwrap import _unwrap_to_base_model
from rsm_pipeline.models.importance import _get_feature_importances

_log = logging.getLogger(__name__)


def _sample_rows(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed).copy()


def apply_explain(
    model: Any,
    sp: SplitResult,
    cfg: RsmConfig,
    feats: list[str],
    target: str,
    run_dir: Path,
) -> dict[str, Any]:
    """Run SHAP + PDP + native importance. Return summary meta."""
    assert cfg.explain is not None
    ex_cfg = cfg.explain
    out_dir = run_dir / ex_cfg.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    base_model = _unwrap_to_base_model(model)
    seed = cfg.run.seed
    top_k = min(ex_cfg.top_k_features, len(feats))

    # 1. native importance (on unwrapped base model)
    native_path: str | None = None
    imp = _get_feature_importances(base_model, feats)
    if imp:
        df_imp = (
            pd.DataFrame(
                {"feature": list(imp.keys()), "importance": list(imp.values())}
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        out_csv = out_dir / "native_importance.csv"
        df_imp.to_csv(out_csv, encoding="utf-8", index=False)
        native_path = str(out_csv.relative_to(run_dir))

    # 2. SHAP
    shap_block: dict[str, Any] = {"kind": None}
    top_features: list[str] = []
    if ex_cfg.shap.enabled:
        X_bg = _sample_rows(sp.train[feats], ex_cfg.shap.n_background, seed)
        X_target = _sample_rows(sp.val[feats], ex_cfg.shap.n_target, seed)
        explainer, kind = _select_explainer(model, X_bg)
        if explainer is not None:
            try:
                raw = explainer.shap_values(X_target)
                sv = _normalize_shap_values(raw)
                # CSV
                csv_path = out_dir / "shap_values.csv"
                df_sv = pd.DataFrame(sv, columns=feats, index=X_target.index)
                df_sv["_score"] = model.predict_proba(X_target)[:, 1]
                df_sv.to_csv(csv_path, encoding="utf-8")
                # Plots
                summary_paths = _save_summary_plots(
                    sv, X_target, out_dir, top_k, run_dir
                )
                # Top features by mean |SHAP|
                mean_abs = np.abs(sv).mean(axis=0)
                idx_top = np.argsort(-mean_abs)[:top_k]
                top_features = [feats[i] for i in idx_top]
                # Local waterfalls
                risk = model.predict_proba(X_target)[:, 1]
                local_paths = _save_local_waterfalls(
                    sv,
                    X_target,
                    _expected_value_for_class1(explainer),
                    feats,
                    risk,
                    ex_cfg.n_local_samples,
                    out_dir,
                    run_dir,
                )
                shap_block = {
                    "kind": kind,
                    "n_target": int(len(X_target)),
                    "n_background": int(len(X_bg)),
                    "values_csv": str(csv_path.relative_to(run_dir)),
                    "summary_plots": summary_paths,
                    "local_plots": local_paths,
                }
            except Exception as exc:  # noqa: BLE001
                shap_block = {
                    "kind": None,
                    "fallback_reason": f"shap run failed: {exc}",
                }
        else:
            shap_block = {"kind": None, "fallback_reason": kind}

    # 3. Top features fallback if SHAP didn't run
    if not top_features:
        if imp:
            top_features = sorted(imp.keys(), key=lambda k: -imp[k])[:top_k]
        else:
            top_features = feats[:top_k]

    # 4. PDP
    pdp_block: dict[str, Any] = {"n_features": 0, "plots": []}
    if ex_cfg.pdp.enabled:
        X_pdp = _sample_rows(sp.val[feats], ex_cfg.pdp.n_samples, seed)
        pdp_paths = _generate_pdp_plots(
            model,
            X_pdp,
            top_features,
            out_dir,
            ex_cfg.pdp.grid_resolution,
            run_dir,
        )
        pdp_block = {"n_features": len(pdp_paths), "plots": pdp_paths}

    summary = {
        "model_class": type(_unwrap_to_base_model(model)).__name__,
        "shap": shap_block,
        "pdp": pdp_block,
        "native_importance_path": native_path,
        "top_features": top_features,
    }
    summary_path = out_dir / "explain_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary
