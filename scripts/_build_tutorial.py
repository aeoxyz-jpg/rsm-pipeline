"""Build notebooks/00_tutorial.ipynb programmatically.

Run as `python3.11 scripts/_build_tutorial.py` from the project root.
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "00_tutorial.ipynb"


def md(*lines: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell("\n".join(lines))


def code(*lines: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell("\n".join(lines))


nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.11"},
}

cells: list = []

# 0 — title
cells.append(
    md(
        "# RSM Pipeline — End-to-End Tutorial",
        "",
        "This notebook walks through the full lifecycle of the **Risk Score ML Pipeline**:",
        "",
        "1. Train a model with `rsm-train`",
        "2. Inspect the per-run artifacts (metrics, decile lift, plots, scorecard, SHAP, etc.)",
        "3. Score new data with the persisted `TrainedBundle` and `rsm-predict`",
        "4. Serve predictions via the FastAPI `rsm-serve` app",
        "5. Compare reference vs current data with `rsm-monitor`",
        "",
        "All commands run on the bundled synthetic dataset (`data/sample/synthetic_binary.csv`).",
        "",
        "**Prereqs:** `pip install -e .` from the project root.",
    )
)

# 1 — setup
cells.append(md("## 1. Setup"))
cells.append(
    code(
        "from pathlib import Path",
        "import json",
        "import joblib",
        "import pandas as pd",
        "import numpy as np",
        "from IPython.display import Image, display",
        "",
        "ROOT = Path.cwd().resolve()",
        "if ROOT.name == 'notebooks':",
        "    ROOT = ROOT.parent",
        "print('project root:', ROOT)",
        "pd.set_option('display.max_columns', 20)",
        "pd.set_option('display.width', 160)",
    )
)

# 2 — peek at synthetic data
cells.append(md("## 2. Peek at the data"))
cells.append(
    code(
        "df = pd.read_csv(ROOT / 'data/sample/synthetic_binary.csv')",
        "print('shape:', df.shape)",
        "print('label rate:', df['label'].mean())",
        "df.head()",
    )
)

# 3 — train
cells.append(
    md(
        "## 3. Train",
        "",
        "Run `rsm-train` end-to-end on `configs/example_config.yaml`. "
        "We invoke the CLI directly via Click's `CliRunner` so this notebook stays "
        "in-process and reproducible.",
    )
)
cells.append(
    code(
        "from click.testing import CliRunner",
        "from rsm_pipeline.cli.train import main as train_main",
        "",
        "runner = CliRunner()",
        "result = runner.invoke(",
        "    train_main,",
        "    ['--config', str(ROOT / 'configs/example_config.yaml')],",
        "    catch_exceptions=False,",
        ")",
        "print('exit code:', result.exit_code)",
        "for line in result.output.splitlines()[-10:]:",
        "    print(line)",
    )
)

# 4 — locate latest run
cells.append(md("## 4. Locate the latest run"))
cells.append(
    code(
        "runs = sorted((ROOT / 'experiments').iterdir(), key=lambda p: p.stat().st_mtime)",
        "latest = runs[-1]",
        "print('run dir:', latest)",
        "print('contents:', sorted(p.name for p in latest.iterdir()))",
    )
)

# 5 — top-level metrics
cells.append(md("## 5. Top-level metrics (`run.json`)"))
cells.append(
    code(
        "rj = json.loads((latest / 'run.json').read_text(encoding='utf-8'))",
        "print('model:', rj['model'])",
        "print()",
        "for fold, m in rj.get('metrics', {}).items():",
        "    flat = {k: round(v, 4) for k, v in m.items() if isinstance(v, (int, float))}",
        "    print(f'{fold:>4}: {flat}')",
    )
)

# 6 — decile lift table
cells.append(md("## 6. Decile-lift table (val)"))
cells.append(
    code(
        "decile = pd.read_csv(latest / 'reports/decile_lift_val.csv')",
        "decile.head(15)",
    )
)

# 7 — plots — ROC, KS, calibration
cells.append(md("## 7. Diagnostic plots"))
cells.append(
    code(
        "for name in ['roc_val.png', 'ks_val.png', 'calibration_compare.png',",
        "             'gain_val.png']:",
        "    p = latest / 'reports/plots' / name",
        "    if p.exists():",
        "        print(name)",
        "        display(Image(filename=str(p)))",
    )
)

# 8 — feature importances + SHAP
cells.append(
    md(
        "## 8. Feature importances + SHAP",
        "",
        "Native importances (LR / tree models) sit next to SHAP-based explanations.",
    )
)
cells.append(
    code(
        "imp_path = latest / 'reports/feature_importances.json'",
        "if imp_path.exists():",
        "    imp = json.loads(imp_path.read_text(encoding='utf-8'))",
        "    print('native importance (top 5):')",
        "    for f, v in sorted(imp.items(), key=lambda kv: -kv[1])[:5]:",
        "        print(f'  {f:>8}: {v:.4f}')",
        "",
        "shap_bar = latest / 'reports/explain/shap_summary_bar.png'",
        "if shap_bar.exists():",
        "    display(Image(filename=str(shap_bar)))",
        "shap_bee = latest / 'reports/explain/shap_summary_beeswarm.png'",
        "if shap_bee.exists():",
        "    display(Image(filename=str(shap_bee)))",
    )
)

# 9 — calibration summary
cells.append(md("## 9. Calibration summary (Platt vs Isotonic)"))
cells.append(
    code(
        "cal_path = latest / 'reports/calibration_summary.json'",
        "if cal_path.exists():",
        "    cs = json.loads(cal_path.read_text(encoding='utf-8'))",
        "    print('kind:', cs['kind'])",
        "    print('brier_pre:', round(cs['brier_pre'], 4))",
        "    print('brier_post:', round(cs['brier_post'], 4))",
        "    print('note:', cs.get('note', ''))",
    )
)

# 10 — scorecard
cells.append(
    md(
        "## 10. Scorecard",
        "",
        "If the model is `LogisticRegression` *and* every feature is WoE-encoded,"
        " the pipeline emits a classic PDO scorecard. Otherwise it falls back to"
        " a rank-score band (`[300, 850]` by default).",
    )
)
cells.append(
    code(
        "sc_summary = latest / 'reports/scorecard_summary.json'",
        "if sc_summary.exists():",
        "    s = json.loads(sc_summary.read_text(encoding='utf-8'))",
        "    print('mode:', s['mode'])",
        "    if s['mode'] == 'pdo':",
        "        print('factor:', round(s['factor'], 3),",
        "              'offset:', round(s['offset'], 3),",
        "              'n_features:', s['n_features'])",
        "        sc_csv = latest / 'reports/scorecard.csv'",
        "        if sc_csv.exists():",
        "            display(pd.read_csv(sc_csv).head(10))",
        "    else:",
        "        print('fallback_reason:', s.get('fallback_reason'))",
    )
)

# 11 — score new data with the bundle
cells.append(
    md(
        "## 11. Score new data with the `TrainedBundle`",
        "",
        "The bundle persists every fitted stage (preprocessing → FE → FS → calibrated"
        " model → optional Scorer). It applies the chain to raw inputs.",
    )
)
cells.append(
    code(
        "bundle = joblib.load(latest / 'artifacts/bundle.joblib')",
        "print('bundle build_meta:', bundle.build_meta)",
        "print('raw input columns:', bundle.raw_input_columns)",
        "print('has scorer:', bundle.scorer is not None)",
        "",
        "from rsm_pipeline.serving.batch import score_batch",
        "scored = score_batch(bundle, df.head(10), threshold=0.5,",
        "                     include_score=bundle.scorer is not None)",
        "scored[['proba_1', 'predict'] +",
        "       (['score'] if 'score' in scored.columns else [])]",
    )
)

# 12 — REST via TestClient
cells.append(
    md(
        "## 12. REST via `rsm-serve` (in-process)",
        "",
        "We use FastAPI's `TestClient` to exercise the API without binding a port."
        " The same `build_app` is what the CLI hands to `uvicorn`.",
    )
)
cells.append(
    code(
        "from fastapi.testclient import TestClient",
        "from rsm_pipeline.serving.api import build_app",
        "",
        "client = TestClient(build_app(bundle))",
        "print('GET /health ->', client.get('/health').json())",
        "version = client.get('/version').json()",
        "print('GET /version keys:', sorted(version.keys()))",
        "",
        "records = df.head(3).to_dict(orient='records')",
        "resp = client.post(",
        "    '/predict',",
        "    json={'records': records, 'threshold': 0.5,",
        "          'include_score': bundle.scorer is not None},",
        ")",
        "print('POST /predict ->', resp.status_code)",
        "for k, v in resp.json().items():",
        "    print(f'  {k}: {v}')",
    )
)

# 13 — monitoring
cells.append(
    md(
        "## 13. Monitoring — reference vs current",
        "",
        "We synthesize a perturbed copy of the current data (shuffle `f0`, shift `f1` by"
        " +1.5) so the report shows non-trivial PSI / CSI values.",
        "",
        "**PSI / CSI tier legend:**",
        "- `< 0.1` — `stable`",
        "- `0.1 ≤ x < 0.25` — `minor_shift`",
        "- `≥ 0.25` — `major_shift`",
    )
)
cells.append(
    code(
        "import tempfile",
        "from rsm_pipeline.monitoring.apply import apply_monitoring",
        "",
        "rng = np.random.default_rng(0)",
        "ref = df.copy()",
        "cur = df.copy()",
        "cur['f0'] = rng.permutation(cur['f0'].values)",
        "cur['f1'] = cur['f1'] + 1.5",
        "",
        "with tempfile.TemporaryDirectory() as td:",
        "    out = Path(td) / 'monitor'",
        "    summary = apply_monitoring(",
        "        bundle, ref, cur, out,",
        "        has_labels=True, target='label',",
        "    )",
        "print('score_psi:', summary['score_psi'])",
        "print('csi:', summary['csi'])",
        "print('quality:', summary['data_quality'])",
        "print('performance:', summary.get('performance'))",
    )
)

# 14 — closing
cells.append(
    md(
        "## 14. Where to go next",
        "",
        "- Edit `configs/example_config.yaml` to swap models (`logreg` → `xgboost`,"
        " `random_forest`, `lightgbm`, `mlp`, `voting`, `stacking`) or tune backends"
        " (`grid` → `random` / `optuna`).",
        "- Wire your own dataset by pointing `data.source.path` at a CSV / Parquet"
        " with a binary `label` column and your features.",
        "- Bundle a deployable artifact with `cfg.export.kinds = ['joblib', 'onnx']`,"
        " then ship via `rsm-predict` (batch) or `rsm-serve` (REST).",
        "- Schedule `rsm-monitor` weekly with the latest scoring batch as `--current`"
        " and the training fold as `--reference` to flag drift.",
    )
)

nb["cells"] = cells

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {NB_PATH}")
