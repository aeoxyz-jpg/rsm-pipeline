# Design Specification — Risk Score ML Pipeline

> Full design specification for this repository. Read this before generating, modifying, or refactoring any code.

---

## Quick Commands

```
Install (editable):    pip install -e ".[dev]"
Format:                black src scripts notebooks
Train:                 rsm-train   --config configs/example_config.yaml
Batch predict:         rsm-predict --model artifacts/model.joblib --input data/new.csv
Serve REST:            rsm-serve   --model artifacts/model.joblib --port 8000
Monitor drift:         rsm-monitor --reference baseline.parquet --current latest.parquet
```

No tests, no linters beyond `black`. The tutorial notebook (`notebooks/00_tutorial.ipynb`) is the de facto smoke test.

---

## 1. Project Overview

A **general-purpose risk scoring machine learning pipeline** for **binary classification** problems (label ∈ {0, 1}). The pipeline is designed for general business risk use cases such as **churn prediction, default prediction, and similar event-occurrence modeling** — not limited to credit/fraud, but informed by traditional credit-risk modeling best practices (WoE/IV, scorecards, KS, PSI, etc.).

The pipeline covers the **full model development lifecycle**:

```
Data Loading → EDA → Preprocessing → Feature Engineering →
Feature Selection → Train/Val/Test Split → Model Training →
Hyperparameter Tuning → Evaluation → Calibration → Scorecard →
Explainability → Export → Batch/REST Scoring → Monitoring
```

**Target users:** multiple data scientists within a team sharing the same codebase.

**Distribution model & runtime environments — CRITICAL CONTEXT:**

This project is **developed on macOS (Apple Silicon)** and **distributed as a zip archive to teammates' work computers**. The work computers run a **mix of operating systems and architectures**:

- Linux on x86_64 (most common)
- Windows on x86_64
- macOS on Intel or Apple Silicon

After unzipping, recipients install with `pip install -e .` (or `pip install .`) inside a fresh virtual environment. The package must therefore be **fully cross-platform and portable**. Specifically:

- **No platform-specific dependencies.** Do not import or add packages that only work on one OS or architecture (e.g. `mlx`, `coremltools`, `pywin32`, `appscript`, `applescript`).
- **No hardcoded absolute paths.** Never write author-specific paths. Always use `pathlib.Path`, `tempfile`, and paths relative to either the project root or a config-supplied location.
- **No POSIX-only shell calls.** Avoid `os.system("ls")`, `subprocess` calls to `bash`, backticks, or shell-builtin assumptions. If shell interaction is unavoidable, branch on `sys.platform` or use Python equivalents (`pathlib`, `shutil`).
- **No architecture-specific acceleration paths.** Do not branch on Metal / MPS / MLX. Pure CPU execution must be the default and only required path.
- **No assumption of internet access at runtime.** Some work computers are on restricted networks. Anything that must download data, models, or weights at runtime must be optional and fail gracefully with a clear message.
- **File encoding & line endings:** always read/write text files with explicit `encoding="utf-8"`. Never rely on platform default encoding (Windows defaults to cp1252, which breaks on non-ASCII content).
- **CLI commands** declared in `pyproject.toml` (`rsm-train`, `rsm-predict`, `rsm-serve`, `rsm-monitor`) are the canonical user entry points and must work identically on all three OSes.

**Driver style:** Notebook-first for exploration and iteration, with supporting scripts/modules for reusable logic. Notebooks call into a clean Python package; they do not contain heavy logic themselves. Notebooks use **relative paths from the notebook file** so they run unchanged on every teammate's machine.

**Language convention:** All code, comments, docstrings, variable names, log messages, and generated reports are in **English**.

---

## 2. Tech Stack

**Required:**
- Python 3.10+
- `pandas`, `numpy`, `scipy`
- `scikit-learn` (primary framework — pipelines, transformers, metrics)
- `xgboost`, `lightgbm`, `catboost`
- `optuna` (Bayesian / TPE hyperparameter tuning)
- `shap` (explainability)
- `matplotlib`, `seaborn`, `plotly` (visualizations)
- `jupyter` / `ipykernel` (notebook delivery)
- `pyarrow` (Parquet IO)
- `joblib` (model persistence)
- `fastapi` + `uvicorn` (REST scoring service)
- `onnx`, `onnxmltools`, `skl2onnx` (model export)
- `black` (formatting — the **only** style tool required)

**Optional / on-demand:**
- `imbalanced-learn` (SMOTE family)
- `category_encoders` (target / WoE encoders as fallback to custom impl)
- `hyperopt` (alternative HPO backend)

Explicitly out of scope: pytest, mypy, ruff, pre-commit, MLflow, W&B, DVC, Docker tooling, Sphinx, MkDocs.

---

## 3. Repository Structure

```
risk-pipeline/
├── DESIGN.md
├── README.md
├── pyproject.toml                # black config, dependencies
├── configs/                      # YAML configs per experiment
│   └── example_config.yaml
├── data/                         # gitignored, raw + processed
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
│   └── 00_tutorial.ipynb         # end-to-end walkthrough
├── src/rsm_pipeline/
│   ├── __init__.py
│   ├── config/                   # config loading / validation
│   ├── data/                     # IO, splitting
│   ├── preprocessing/            # missing/outlier handling
│   ├── feature_engineering/      # WoE/IV, binning, encoders, scalers
│   ├── feature_selection/        # feature selection methods
│   ├── imbalance/                # imbalance handling
│   ├── models/                   # model factory + wrappers
│   ├── tuning/                   # HPO backends
│   ├── evaluation/               # metrics, decile tables, plots
│   ├── calibration/              # Platt / Isotonic
│   ├── scorecard/                # PDO scorecard generation
│   ├── explain/                  # SHAP, PDP, importance
│   ├── export/                   # joblib, ONNX, PMML
│   ├── serving/                  # FastAPI app + batch scorer
│   ├── monitoring/               # PSI, CSI, performance drift
│   ├── tracking/                 # local JSON/SQLite experiment log
│   └── cli/                      # CLI entry points
├── scripts/                      # thin CLI wrappers
├── experiments/                  # tracking DB + run artifacts (gitignored)
└── reports/                      # generated JSON/CSV outputs (gitignored)
```

Keep all reusable logic in `src/`. Notebooks **import from `src/`**, never duplicate logic.

---

## 4. Functional Requirements

### 4.1 Data Layer

- **Input formats:** Local CSV and Parquet files. Use `pyarrow` for Parquet.
- **Scale:** Single-machine, < 1M rows. No Spark / Dask / Ray.
- **Splitting strategies (both supported, user-selectable via config):**
  - **Random split** — stratified by label, configurable train/val/test ratios.
  - **Time-based split** — strict temporal ordering by a user-specified date column, with no future leakage. Train/val/test boundaries defined by date cutoffs or proportional time slicing.
- A thin `DataLoader` abstraction so adding new sources later is one new subclass.

### 4.2 Preprocessing — Missing & Outlier Handling

Class imbalance is common in risk data. Missing-value handling is critical and must be **leakage-safe** (fit on train only, transform on val/test).

Imputation strategies (user-selectable, **per-column overrideable**):
- Mean / median / mode
- KNN imputation
- Indicator variable (add `<col>_is_missing` flag)
- **Custom fill values** — user can specify sentinel values per column (e.g. `99`, `98`, `97`, `0`, `-1`). First-class, not an afterthought. Config schema accepts a `{column: fill_value}` map.

Outlier handling: IQR clipping, percentile clipping (e.g. winsorize at 1%/99%), z-score capping. All optional and per-column configurable.

All transformers are sklearn-compatible (`fit`, `transform`, `fit_transform`) so they slot into `sklearn.pipeline.Pipeline`.

### 4.3 Feature Engineering

- Numerical and categorical features are both first-class citizens.
- **WoE (Weight of Evidence) and IV (Information Value)** — non-negotiable for the risk-modeling use case.
  - A `WoEEncoder` for categorical features and a `WoEBinningEncoder` for numerical features (with monotonic / chi-merge / quantile binning options).
  - IV computed per feature and exposed as a fitted attribute.
- Standard encoders: One-Hot, Ordinal, Target (mean) encoding — all leakage-safe.
- Standard scalers: StandardScaler, MinMaxScaler, RobustScaler.
- All feature engineering composable via sklearn `Pipeline` / `ColumnTransformer`.

### 4.4 Feature Selection

User-selectable and stackable:
- **IV threshold filter** (typical cutoffs: drop IV < 0.02; consider strong if IV > 0.3)
- **Correlation filter** (drop one of any pair with |ρ| above threshold)
- **Variance filter** (drop near-zero-variance features)
- **Model-based importance** (fit a tree model, threshold by importance percentile)
- **Recursive Feature Elimination (RFE / RFECV)**

Output a feature-selection report listing what was dropped and why.

### 4.5 Imbalance Handling

User-selectable strategies:
- `class_weight` (passed through to estimators that support it)
- Random oversampling
- Random undersampling
- SMOTE (and SMOTE-NC for mixed types)
- ADASYN
- Combined (SMOTEENN, SMOTETomek)
- "auto" mode: choose a sensible default based on observed imbalance ratio

**Sampling is applied only to the training fold** — never to validation or test data. Enforced in code.

### 4.6 Models

A unified factory interface for:
- Logistic Regression (interpretable baseline — first-class for scorecards)
- XGBoost
- LightGBM
- CatBoost
- Random Forest
- Extra Trees
- MLP (simple feedforward, via `sklearn.neural_network.MLPClassifier`)
- Ensembles: `VotingClassifier`, `StackingClassifier`

Each model wrapper exposes a uniform interface: `fit`, `predict`, `predict_proba`, `feature_importances_` (or SHAP fallback), and serialization hooks.

### 4.7 Hyperparameter Tuning

Multiple backends, user-selectable:
- `GridSearchCV`
- `RandomizedSearchCV`
- **Optuna** (default recommendation — TPE sampler with pruning)

A unified `Tuner` API wraps all three so switching is a config change.

### 4.8 Evaluation

**Metrics (all computed and reported):**
- AUC-ROC
- KS statistic
- Gini coefficient
- PR-AUC
- F1, Precision, Recall (at default 0.5 threshold and at user-specified thresholds)
- Brier score
- Log loss

**Decile / percentile lift table:**
- Sort predictions descending by score.
- Bucket the **top 10% into 1%-wide bins** (10 bins covering the top decile), and the remaining 90% into deciles (or percentiles if configured).
- For each bin, report:
  - bin range and cumulative population %
  - **# positives, # negatives** in bin
  - **TPR (cumulative recall), FPR (cumulative)**
  - **Precision, Recall** (cumulative)
  - lift, cumulative lift, cumulative gain
- Output as both a DataFrame (CSV-exportable) and a chart.

**Plots:**
- ROC curve, PR curve
- KS plot
- Confusion matrix (at chosen threshold)
- Calibration curve (reliability diagram)
- Cumulative gain & lift charts
- Score distribution by class

### 4.9 Calibration

Both supported:
- Platt scaling (sigmoid)
- Isotonic regression

Calibration is fit on a held-out validation fold, never on training data. Pre- and post-calibration Brier scores and reliability curves are reported side by side.

### 4.10 Scorecard

A **standard PDO-style scorecard** when the underlying model is Logistic Regression on WoE-encoded features:
- Inputs: `base_score` (default 600), `base_odds` (default 50:1), `pdo` (default 20).
- Output: per-feature, per-bin point assignments — exportable as CSV/JSON.
- Both a `predict_proba` interface and a `predict_score` interface returning integer scores.
- For non-LR models, a `rank_score` (monotonic transform of probability into a 300–850-style integer band) so downstream business users always have an integer score available.

### 4.11 Explainability

- **SHAP — full coverage:** TreeExplainer / LinearExplainer / KernelExplainer fallback, with global summary plots, local force/waterfall plots, and interaction values.
- **Feature importance** (model-native).
- **Partial Dependence Plots** and ICE plots.
- LIME is **not** required.

All explainability outputs are saveable to disk (PNG + underlying CSV/JSON).

### 4.12 Model Export & Deployment

Multiple export targets, all selectable:
- **`joblib` / `pickle`** — full sklearn pipeline including preprocessing.
- **ONNX** — via `skl2onnx` / `onnxmltools` (cover sklearn, XGBoost, LightGBM where supported).
- **PMML** — via `sklearn2pmml` (best-effort; document any model types it cannot handle).
- **Batch scoring** — reads a CSV/Parquet file, applies the saved pipeline, writes scored output. CLI entry point `rsm-predict`.
- **REST API** — FastAPI app exposing `/predict` and `/predict_batch` endpoints with Pydantic request/response schemas. CLI entry point `rsm-serve`. Includes health check and model-version endpoints.

Docker image generation is **out of scope**.

### 4.13 Experiment Tracking

Lightweight, **self-built** — no MLflow / W&B / DVC.
- Backend: SQLite (preferred) or JSON files under `experiments/`.
- Each run records: timestamp, config snapshot (hash + full YAML), git commit (if available), dataset fingerprint (row count + column hash), metrics, artifact paths, tuning history.
- A small query helper to list / compare runs (by metric, by date).

### 4.14 Monitoring

Post-deployment monitoring utilities (callable on a fresh batch of scored data vs. a reference dataset):
- **PSI** (Population Stability Index) on the score distribution.
- **CSI** (Characteristic Stability Index) per feature.
- **Performance drift** — recompute AUC / KS / decile-lift on labeled production data and compare against training baseline.
- **Data quality checks** — distribution shifts, missing-rate changes, new categorical levels.
- Output: structured JSON + CSV reports.

Automatic retraining triggers are **out of scope** — the pipeline only flags drift and reports thresholds; retraining is a human decision.

### 4.15 Reporting

Outputs are kept lightweight:
- **Notebook delivery** is the primary medium for analysis and storytelling.
- **JSON + CSV result files** under `reports/` are the machine-readable output of every run.

No HTML, Markdown, or PDF report generators.

---

## 5. Engineering Conventions

### 5.1 Code Style
- Format every file with **`black`** (default line length 88). No other linters or type checkers.
- Use **type hints throughout** — function signatures, class attributes, public APIs. Hints serve as documentation.
- Prefer pure functions and sklearn-style transformer classes over stateful god-objects.
- Public APIs are stable; internal helpers are marked with a leading underscore.

### 5.2 Documentation
- **Every public function and class has a docstring** in NumPy or Google style, including a one-line summary, parameter descriptions with types, a return description, and an `Examples` section with a runnable snippet.
- `README.md` covers: install, quickstart, config schema, project layout, contribution notes.
- `notebooks/00_tutorial.ipynb` walks a new DS through the full pipeline end-to-end on a sample dataset.

### 5.3 Configuration
- All experiments are driven by YAML configs in `configs/`. Code reads config, never hardcodes hyperparameters or paths.
- A documented `configs/example_config.yaml` covers every section (data, preprocessing, features, sampling, model, tuning, evaluation, export, monitoring).
- Configs are validated at load time with a clear schema (`pydantic`).

### 5.4 Reproducibility
- Every run accepts a `random_state` / `seed` and threads it through all stochastic components (splits, samplers, models, tuners).
- The seed, library versions, and config hash are logged with every run.

### 5.5 Logging
- Use `logging` (stdlib), not `print`. Default level INFO; DEBUG for development.
- Log key checkpoints: data shape after each step, fit/transform timing, metric summaries, artifact paths.

### 5.6 Testing
- Unit tests are not part of this project. The tutorial notebook serves as the de facto smoke test.

### 5.7 Git Hygiene
- `.gitignore` excludes: `data/`, `experiments/`, `reports/`, `*.pkl`, `*.joblib`, `*.onnx`, `.ipynb_checkpoints/`, `__pycache__/`, `.venv/`.

### 5.8 Cross-Platform Compatibility (hard requirement)

Authored on macOS (Apple Silicon) and shipped to teammates running Linux x86_64, Windows x86_64, and macOS (Intel + Apple Silicon). Every file must run unchanged on all three.

**Filesystem & paths**
- Use `pathlib.Path` for all path manipulation. Never use string concatenation or `os.path.join` with hardcoded separators.
- Treat the project root as `Path(__file__).resolve().parents[N]` from a known anchor module, or as the current working directory only when explicitly documented.
- Use `tempfile.gettempdir()` for scratch space, never a hardcoded `/tmp`.
- Never write paths that contain an author-specific home-directory fragment.

**Subprocess & shell**
- Prefer Python-native equivalents: `shutil.copy`, `shutil.rmtree`, `pathlib.Path.glob`, etc.
- If `subprocess` is truly necessary, pass arguments as a list (not a string), do not set `shell=True`, and verify the executable exists on PATH first.
- Do not invoke `bash`, `zsh`, `make`, or shell scripts as part of the runtime pipeline.

**Encoding**
- Always pass `encoding="utf-8"` to `open()`, `Path.read_text()`, `Path.write_text()`, `pd.read_csv()`, `json.dump()` etc. when writing or reading text.

**Dependencies**
- Do not add a dependency without confirming it ships wheels for `cp311` and `cp312` on Linux x86_64, Windows x86_64, macOS x86_64, and macOS arm64.
- Anything that requires a system library (Java for PMML, CUDA for GPU, etc.) belongs in `[project.optional-dependencies]`, not the core list.
- The base `pip install .` must succeed on a clean machine of every supported OS without any prior system-level setup beyond Python itself.

**Determinism across platforms**
- Random seed threading (§5.4) must produce metric values that are *close* across platforms. Floating-point bit-exactness across OS/BLAS implementations is not required, but order-of-magnitude differences indicate a bug.

---

## 6. Engineering Principles

1. **Plan before coding.** For non-trivial changes, outline the approach before writing files.
2. **Respect the layered structure.** Reusable logic goes in `src/rsm_pipeline/<submodule>/`; notebooks and scripts only orchestrate.
3. **Sklearn-compatibility is a hard requirement** for any transformer or estimator. They must work inside `Pipeline` and `ColumnTransformer`, support `fit` / `transform` / `fit_transform`, and be `joblib`-serializable.
4. **Leakage prevention is a hard requirement.** Anything stateful (imputation values, scaler stats, WoE bins, encoders, samplers) is fit on training data only.
5. **Format with `black`** before considering a task done.
6. **Keep `README.md` and the relevant notebook current** whenever a public API changes.
7. **English only** in all generated content.
8. **Cross-platform first.** Before writing any file I/O, subprocess call, or path manipulation, verify the code runs on Linux, Windows, and macOS. When unsure whether an API is portable, choose the more conservative option (`pathlib` over `os.path`, `shutil` over shell commands).
9. **Self-contained.** The package runs with zero external service dependencies on a fresh laptop with only Python + the wheels declared in `pyproject.toml`.
10. **When in doubt about a design choice**, prefer the option that is more explicit, more configurable, and more inspectable — this is a shared team codebase, not a one-off script.

---

## 7. Out of Scope (do not build)

- Distributed compute (Spark, Dask, Ray)
- Cloud-specific data connectors (Snowflake, BigQuery, Redshift, S3)
- MLflow, Weights & Biases, DVC
- Docker images, Kubernetes manifests, CI/CD pipelines
- HTML / PDF / Markdown report generators
- Unit test suites
- LIME, ELI5, alibi, or other explainability libraries beyond SHAP + sklearn-native
- Automatic model retraining triggers
- Multi-class or regression support — this pipeline is **binary classification only**
