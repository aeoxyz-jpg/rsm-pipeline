"""``RunRecorder`` — atomically rewrites ``run.json`` after every stage."""

from __future__ import annotations

import datetime as _dt
import importlib.metadata as _md
import json
import logging
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import yaml

from rsm_pipeline.config.schema import RsmConfig
from rsm_pipeline.paths import project_root
from rsm_pipeline.tracking.fingerprint import config_hash, git_info
from rsm_pipeline.tracking.ids import new_run_id

_log = logging.getLogger(__name__)

_TRACKED_LIBS = (
    "rsm_pipeline",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "pyarrow",
    "pydantic",
    "click",
    "pyyaml",
)


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _library_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in _TRACKED_LIBS:
        try:
            out[name] = _md.version(name)
        except _md.PackageNotFoundError:
            out[name] = "unknown"
    return out


def _set_dotted(d: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = d
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


class RunRecorder:
    """Context manager that owns one run directory and its ``run.json``."""

    def __init__(self, cfg: RsmConfig, config_path: str | Path | None = None):
        self._cfg = cfg
        self._cfg_path = str(config_path) if config_path is not None else None
        self.run_id: str = new_run_id()
        output_root = Path(cfg.run.output_root)
        self._run_dir: Path = output_root / self.run_id
        self._record: dict[str, Any] = {}
        self._t0: float | None = None

    # -- lifecycle ----------------------------------------------------

    def __enter__(self) -> "RunRecorder":
        self._run_dir.mkdir(parents=True, exist_ok=True)
        (self._run_dir / "splits").mkdir(exist_ok=True)
        (self._run_dir / "artifacts").mkdir(exist_ok=True)

        cfg_dump = self._cfg.model_dump(mode="json")
        (self._run_dir / "config.snapshot.yaml").write_text(
            yaml.safe_dump(cfg_dump, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        commit, dirty = git_info(project_root())
        self._t0 = time.monotonic()
        self._record = {
            "run_id": self.run_id,
            "name": self._cfg.run.name,
            "status": "running",
            "started_at": _utc_now_iso(),
            "finished_at": None,
            "duration_seconds": None,
            "seed": self._cfg.run.seed,
            "git_commit": commit,
            "git_dirty": dirty,
            "python_version": sys.version.split()[0],
            "platform": f"{platform.system().lower()}-{platform.machine().lower()}",
            "library_versions": _library_versions(),
            "config_path": self._cfg_path,
            "config_hash": config_hash(cfg_dump),
            "dataset_fingerprint": None,
            "split": None,
            "model": None,
            "metrics": {},
            "artifacts": {},
            "errors": None,
        }
        self._write()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        finished = _utc_now_iso()
        duration = round(time.monotonic() - (self._t0 or time.monotonic()), 3)
        if exc is None:
            self._record["status"] = "succeeded"
        else:
            self._record["status"] = "failed"
            self._record["errors"] = {
                "type": exc_type.__name__ if exc_type else "Unknown",
                "message": str(exc),
                "traceback": "".join(traceback.format_exception(exc_type, exc, tb))[
                    :8000
                ],
            }
        self._record["finished_at"] = finished
        self._record["duration_seconds"] = duration
        self._write()
        return False  # never suppress

    # -- public API ---------------------------------------------------

    def run_dir(self) -> Path:
        return self._run_dir

    def log_meta(self, **kv: Any) -> None:
        self._record.update(kv)
        self._write()

    def log_dataset_fingerprint(self, fp: dict[str, Any]) -> None:
        self._record["dataset_fingerprint"] = fp
        self._write()

    def log_split(self, split_record: dict[str, Any]) -> None:
        self._record["split"] = split_record
        self._write()

    def log_model(self, kind: str, params: dict[str, Any]) -> None:
        self._record["model"] = {"kind": kind, "params": params}
        self._write()

    def log_metrics(self, fold: str, metrics: dict[str, float]) -> None:
        self._record.setdefault("metrics", {})[fold] = metrics
        self._write()

    def log_artifact(self, dotted_key: str, rel_path: str) -> None:
        artifacts = self._record.setdefault("artifacts", {})
        _set_dotted(artifacts, dotted_key, rel_path)
        self._write()

    # -- internals ----------------------------------------------------

    def _write(self) -> None:
        target = self._run_dir / "run.json"
        tmp = target.with_suffix(".json.tmp")
        try:
            tmp.write_text(
                json.dumps(self._record, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            tmp.replace(target)
        except OSError as e:
            _log.warning("RunRecorder: failed to write run.json: %s", e)


# --- helpers ---------------------------------------------------------


def load_run(run_dir: Path) -> dict[str, Any]:
    """Read ``run_dir/run.json`` and return it as a dict."""
    return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))


def list_runs(output_root: Path) -> list[dict[str, Any]]:
    """Return all ``run.json`` records under ``output_root`` sorted desc by ``started_at``."""
    if not output_root.exists():
        return []
    out: list[dict[str, Any]] = []
    for child in output_root.iterdir():
        rj = child / "run.json"
        if rj.is_file():
            try:
                out.append(json.loads(rj.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError) as e:
                _log.warning("list_runs: skipping %s: %s", rj, e)
    out.sort(key=lambda r: r.get("started_at") or "", reverse=True)
    return out
