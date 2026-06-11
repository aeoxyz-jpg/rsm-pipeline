"""``rsm-train`` — orchestrator.

Pipeline: load_config → load_dataset → split → preprocess → feature_engineer →
feature_select → imbalance → fit_model → evaluate → persist → record run.
Heavy logic lives in :mod:`rsm_pipeline`; this module is a thin glue layer.
"""

from __future__ import annotations

import logging
import sys

import json

import click

from rsm_pipeline.config import load_config
from rsm_pipeline.data import load_dataset, split_dataset
from rsm_pipeline.feature_engineering import apply_feature_engineering
from rsm_pipeline.feature_selection import apply_feature_selection
from rsm_pipeline.logging_setup import configure_logging
from rsm_pipeline.models import (
    _get_feature_importances,
    fit_model,
    persist_model,
)
from rsm_pipeline.evaluation import apply_full_evaluation
from rsm_pipeline.calibration import apply_calibration
from rsm_pipeline.scorecard import apply_scorecard
from rsm_pipeline.explain import apply_explain
from rsm_pipeline.models.apply import fit_model_with_estimator
from rsm_pipeline.tuning import apply_tuning
from rsm_pipeline.imbalance import apply_imbalance
from rsm_pipeline.preprocessing import apply_preprocessing
from rsm_pipeline.tracking import RunRecorder, fingerprint_dataset
from rsm_pipeline.export import apply_export


def _resolve_date_column(cfg) -> str | None:
    if cfg.split.method == "time":
        if cfg.split.time and cfg.split.time.date_column:
            return cfg.split.time.date_column
        return cfg.data.date_column
    return cfg.data.date_column


@click.command(name="rsm-train")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the YAML config file.",
)
def main(config_path: str) -> None:
    """Run a single training pipeline end-to-end."""
    cfg = load_config(config_path)

    with RunRecorder(cfg, config_path=config_path) as run:
        configure_logging(level="INFO", run_dir=run.run_dir())
        log = logging.getLogger("rsm_pipeline.cli.train")
        log.info(
            "run_id=%s name=%s seed=%d output_dir=%s",
            run.run_id,
            cfg.run.name,
            cfg.run.seed,
            run.run_dir(),
        )

        # 1. Load
        df = load_dataset(cfg)
        run.log_dataset_fingerprint(fingerprint_dataset(df, cfg.data.source.path))

        # 2. Split
        sp = split_dataset(df, cfg)
        for fold_name, fold_df in (
            ("train", sp.train),
            ("val", sp.val),
            ("test", sp.test),
        ):
            rel = f"splits/{fold_name}.parquet"
            (run.run_dir() / "splits").mkdir(parents=True, exist_ok=True)
            fold_df.to_parquet(run.run_dir() / rel, engine="pyarrow", index=False)
            run.log_artifact(f"splits.{fold_name}", rel)
        run.log_split(sp.meta)

        # 3. Build feature list
        target = cfg.data.target.column
        date_col = _resolve_date_column(cfg)
        feats = [c for c in sp.train.columns if c != target and c != date_col]
        raw_input_columns = list(feats)  # snapshot before any pipeline stage transforms
        log.info("feature columns: n=%d (%s)", len(feats), feats)

        # 3.5. Preprocess (optional)
        pre = None
        if cfg.preprocessing is not None:
            sp, pre, pre_meta = apply_preprocessing(sp, cfg, feats)
            run.log_meta(preprocessing=pre_meta)
            for fold_name, fold_df in (
                ("train", sp.train),
                ("val", sp.val),
                ("test", sp.test),
            ):
                rel = f"splits/{fold_name}.parquet"
                fold_df.to_parquet(run.run_dir() / rel, engine="pyarrow", index=False)
            feats = [c for c in sp.train.columns if c != target and c != date_col]
            log.info(
                "preprocessed: %d input cols -> %d output cols",
                pre_meta["n_input_cols"],
                len(feats),
            )

        # 3.6. Feature engineering (optional)
        fe = None
        if cfg.feature_engineering is not None:
            sp, fe, fe_meta = apply_feature_engineering(sp, cfg, feats, target)
            run.log_meta(feature_engineering=fe_meta)
            feats = [c for c in sp.train.columns if c != target and c != date_col]
            log.info(
                "feature_engineering: %d input cols -> %d output cols",
                fe_meta["n_input_cols"],
                fe_meta["n_output_cols"],
            )

        # 3.7. Feature selection (optional)
        fs = None
        if cfg.feature_selection is not None:
            sp, fs, fs_meta = apply_feature_selection(
                sp, cfg, feats, target, run.run_dir()
            )
            run.log_meta(feature_selection=fs_meta)
            for fold_name, fold_df in (
                ("train", sp.train),
                ("val", sp.val),
                ("test", sp.test),
            ):
                rel = f"splits/{fold_name}.parquet"
                fold_df.to_parquet(run.run_dir() / rel, engine="pyarrow", index=False)
            feats = [c for c in sp.train.columns if c != target and c != date_col]
            log.info(
                "feature_selected: %d input cols -> %d output cols",
                fs_meta["n_input_cols"],
                len(feats),
            )

        # 3.8. Imbalance handling (optional)
        if cfg.imbalance is not None:
            sp, imb_meta = apply_imbalance(sp, cfg, feats, target, run.run_dir())
            run.log_meta(imbalance=imb_meta)
            # Only train.parquet may have changed; val/test unchanged.
            sp.train.to_parquet(
                run.run_dir() / "splits/train.parquet",
                engine="pyarrow",
                index=False,
            )
            log.info(
                "imbalance: kind=%s before=%s after=%s",
                imb_meta["kind"],
                imb_meta["before"],
                imb_meta["after"],
            )

        # Resolve imbalance class_weight once (used by tuning + fit)
        imb_class_weight = None
        if cfg.imbalance is not None:
            cw = imb_meta.get("class_weight")
            imb_class_weight = cw if cw else None

        # 3.9. Hyperparameter tuning (optional)
        best_params: dict | None = None
        if cfg.tuning is not None:
            best_params, tune_meta = apply_tuning(
                cfg,
                sp.train[feats],
                sp.train[target],
                class_weight=imb_class_weight,
                run_dir=run.run_dir(),
            )
            run.log_meta(tuning=tune_meta)
            log.info(
                "tuning: backend=%s n_trials=%d best_score=%.4f best_params=%s",
                tune_meta["backend"],
                tune_meta["n_trials"],
                tune_meta["best_score"],
                best_params,
            )

        # 4. Fit (with merged best params if tuning ran)
        if best_params is not None:
            estimator_cfg_dict = cfg.model.estimator.model_dump()
            estimator_cfg_dict.update(best_params)
            estimator_cfg = type(cfg.model.estimator)(**estimator_cfg_dict)
            model = fit_model_with_estimator(
                cfg,
                estimator_cfg,
                sp.train[feats],
                sp.train[target],
                class_weight=imb_class_weight,
            )
        else:
            model = fit_model(
                cfg,
                sp.train[feats],
                sp.train[target],
                class_weight=imb_class_weight,
            )
        rel = persist_model(model, run.run_dir(), cfg)
        run.log_model(
            cfg.model.estimator.kind,
            cfg.model.estimator.model_dump(exclude={"kind"}),
        )
        run.log_artifact("model.joblib", rel)

        # 4.5. Importances (best-effort)
        imp = _get_feature_importances(model, feats)
        if imp:
            (run.run_dir() / "reports").mkdir(parents=True, exist_ok=True)
            (run.run_dir() / "reports/feature_importances.json").write_text(
                json.dumps(imp, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        # 4.7. Calibration (optional)
        if cfg.calibration is not None and cfg.calibration.kind != "none":
            model, cal_meta = apply_calibration(
                model,
                sp,
                cfg,
                feats,
                target,
                run.run_dir(),
            )
            run.log_meta(calibration=cal_meta)
            log.info(
                "calibration: kind=%s brier_pre=%.4f brier_post=%.4f",
                cal_meta["kind"],
                cal_meta["brier_pre"],
                cal_meta["brier_post"],
            )
            rel = persist_model(model, run.run_dir(), cfg)
            run.log_artifact("model.joblib", rel)

        # 5. Full evaluation
        metrics_by_fold = apply_full_evaluation(
            model,
            sp,
            cfg,
            feats,
            target,
            run.run_dir(),
        )
        for fold_name, m in metrics_by_fold.items():
            run.log_metrics(fold_name, m)
            log.info(
                "metrics fold=%s -> roc_auc=%.4f ks=%.4f gini=%.4f brier=%.4f",
                fold_name,
                m["roc_auc"],
                m["ks"],
                m["gini"],
                m["brier"],
            )

        # 5.5. Scorecard (optional)
        if cfg.scorecard is not None:
            scorer, sc_meta = apply_scorecard(
                model,
                fe,
                sp,
                cfg,
                feats,
                target,
                run.run_dir(),
            )
            run.log_meta(scorecard=sc_meta)
            run.log_artifact("scorer.joblib", sc_meta["artifact_path"])
            log.info(
                "scorecard: mode=%s n_features=%s",
                sc_meta["mode"],
                sc_meta.get("n_features", "n/a"),
            )

        # 5.6. Explain (optional)
        if cfg.explain is not None:
            ex_meta = apply_explain(
                model,
                sp,
                cfg,
                feats,
                target,
                run.run_dir(),
            )
            run.log_meta(explain=ex_meta)
            log.info(
                "explain: shap=%s pdp_n=%d native_importance=%s",
                ex_meta.get("shap", {}).get("kind") or "n/a",
                ex_meta.get("pdp", {}).get("n_features", 0),
                "yes" if ex_meta.get("native_importance_path") else "no",
            )

        # 6. Export
        if cfg.export is not None:
            scorer_obj = None
            if cfg.scorecard is not None:
                # `scorer` was bound inside the scorecard branch
                scorer_obj = locals().get("scorer")
            ex_meta = apply_export(
                model,
                scorer_obj,
                pre,
                fe,
                fs,
                sp,
                cfg,
                feats,
                raw_input_columns,
                target,
                run.run_dir(),
            )
            run.log_meta(export=ex_meta)
            run.log_artifact("bundle.joblib", ex_meta["bundle_path"])
            log.info(
                "export: kinds=%s onnx=%s",
                ex_meta["kinds"],
                ex_meta["onnx"]["status"],
            )

        log.info("run %s succeeded", run.run_id)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
