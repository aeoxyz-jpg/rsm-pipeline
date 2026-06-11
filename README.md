# RSM Pipeline — Risk Score Modeling

A general-purpose, **config-driven machine learning pipeline for binary risk classification** —
churn, default, fraud, and similar event-occurrence problems. It packages the full
credit-risk modeling workflow (WoE/IV, scorecards, KS, PSI/CSI) behind a clean, sklearn-compatible,
cross-platform Python package with CLI entry points and a REST scoring service.

> Built to be unzipped and `pip install`-ed on a teammate's laptop — Linux, Windows, or macOS,
> x86_64 or Apple Silicon — with zero system-level setup beyond Python. See [`DESIGN.md`](DESIGN.md)
> for the complete design specification.

---

## Highlights

- **End-to-end lifecycle** — data loading → preprocessing → feature engineering → selection →
  imbalance handling → model training → tuning → evaluation → calibration → scorecard →
  explainability → export → batch/REST scoring → monitoring.
- **Everything is a YAML config.** One `example_config.yaml` drives an entire run; no hardcoded
  hyperparameters or paths.
- **Leakage-safe by construction.** Every stateful transformer (imputers, scalers, WoE bins,
  encoders, samplers) fits on the training fold only. Sampling never touches val/test.
- **Risk-domain first-class features** — `WoEEncoder` / `WoEBinningEncoder` with monotonic,
  chi-merge, and quantile binning; per-feature IV; PDO scorecards; KS / Gini / decile-lift tables.
- **Cross-platform & offline.** `pathlib` everywhere, explicit UTF-8 I/O, no shell-outs, no
  platform-specific deps, no runtime downloads.
- **Self-built experiment tracking** — every run snapshots its config, seed, dataset fingerprint,
  metrics, and artifacts under `experiments/`.

## Pipeline coverage

| Stage | What's included |
|---|---|
| **Data** | CSV / Parquet loading; random, stratified-random, and time-based splits (no future leakage) |
| **Preprocessing** | mean/median/mode, KNN, missing-indicator, and per-column custom sentinel imputation; IQR / percentile / z-score outlier handling |
| **Feature engineering** | WoE/IV (categorical + numerical binning), One-Hot / Ordinal / Target encoding, Standard / MinMax / Robust scalers |
| **Feature selection** | IV threshold, correlation filter, variance filter, model-based importance, RFE/RFECV — stackable, with a drop report |
| **Imbalance** | class weights, random over/under-sampling, SMOTE / SMOTE-NC, ADASYN, SMOTEENN / SMOTETomek, `auto` mode |
| **Models** | Logistic Regression, XGBoost, LightGBM, CatBoost, Random Forest, Extra Trees, MLP, Voting / Stacking ensembles |
| **Tuning** | GridSearchCV, RandomizedSearchCV, Optuna (TPE + pruning) behind one `Tuner` API |
| **Evaluation** | AUC-ROC, KS, Gini, PR-AUC, F1/precision/recall, Brier, log-loss; decile/percentile lift tables; ROC/PR/KS/calibration/gain/lift plots |
| **Calibration** | Platt and isotonic, fit on held-out validation, with before/after Brier + reliability curves |
| **Scorecard** | standard PDO scorecard (base score / odds / PDO) for LR-on-WoE; integer `rank_score` for any model |
| **Explainability** | SHAP (tree / linear / kernel fallback), native importances, PDP / ICE — all saveable to PNG + CSV/JSON |
| **Export** | `joblib`, ONNX (`skl2onnx` / `onnxmltools`), PMML (best-effort) |
| **Serving** | FastAPI `/predict` + `/predict_batch` with Pydantic schemas, health and model-version endpoints; batch scorer |
| **Monitoring** | PSI on scores, per-feature CSI, performance drift (AUC/KS/lift vs baseline), data-quality checks → JSON + CSV |

## Install

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Python 3.10+ (developed on 3.11). The base install is pure-wheel and works offline on Linux,
Windows, and macOS (Intel + Apple Silicon).

## Quickstart

```bash
# Train end-to-end from a config; writes a full run record under experiments/<run_id>/
rsm-train   --config configs/example_config.yaml

# Score a new file with a saved pipeline
rsm-predict --model artifacts/model.joblib --input data/new.csv --output scored.csv

# Serve the model as a REST API
rsm-serve   --model artifacts/model.joblib --port 8000

# Compare a fresh batch against a reference for drift (PSI / CSI / performance)
rsm-monitor --reference baseline.parquet --current latest.parquet
```

Or walk the whole thing interactively in **[`notebooks/00_tutorial.ipynb`](notebooks/00_tutorial.ipynb)**,
which runs the full pipeline on the bundled 500-row sample dataset (`data/sample/`).

## Configuration

A run is fully described by a YAML file. `configs/example_config.yaml` documents every section:

```yaml
run:        { name: ..., seed: 42, output_root: experiments }
data:       { source: {format, path, csv_options}, target: {column, positive_class}, date_column }
split:      { method: stratified_random | random | time, ... }
preprocessing:       { imputation, outliers, ... }      # per-column overrideable
feature_engineering: { woe, encoders, scalers }
feature_selection:   { iv_threshold, correlation, variance, model_importance, rfe }
imbalance:  { strategy: auto | smote | class_weight | ... }
model:      { kind: logistic_regression | xgboost | lightgbm | ... , params }
tuning:     { backend: optuna | grid | random, ... }
evaluation: { thresholds, decile_table, plots }
calibration:{ method: platt | isotonic }
scorecard:  { base_score: 600, base_odds: 50, pdo: 20 }
explain:    { shap, pdp, importance }
export:     { joblib, onnx, pmml }
```

Code reads config; it never hardcodes hyperparameters or paths.

## Project layout

```
src/rsm_pipeline/      # all reusable logic, by stage (data, preprocessing, feature_engineering, …)
  cli/                 # rsm-train / rsm-predict / rsm-serve / rsm-monitor entry points
configs/               # YAML run configs
notebooks/             # 00_tutorial.ipynb — end-to-end walkthrough
data/sample/           # bundled synthetic dataset for the tutorial
scripts/               # thin CLI wrappers
DESIGN.md              # full design specification
```

Notebooks import from `src/`; they never duplicate logic.

## Design notes

- **sklearn-compatible everywhere** — every transformer/estimator slots into `Pipeline` /
  `ColumnTransformer`, supports `fit` / `transform` / `fit_transform`, and is `joblib`-serializable.
- **Reproducible** — one `seed` threads through splits, samplers, models, and tuners; the seed,
  library versions, and config hash are recorded with every run.
- **Formatted with `black`** (line length 88) — the only style tool used.

Tech stack: pandas / numpy / scipy, scikit-learn, xgboost / lightgbm / catboost, optuna, shap,
matplotlib / seaborn / plotly, pyarrow, fastapi + uvicorn, onnx tooling.

## License

[MIT](LICENSE) © 2026 Nikko & Co
