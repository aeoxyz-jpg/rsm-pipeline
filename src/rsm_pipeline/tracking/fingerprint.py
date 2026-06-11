"""Dataset / config / git fingerprints for run records."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

_log = logging.getLogger(__name__)


def _sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint_dataset(df: pd.DataFrame, source_path: str | Path) -> dict[str, Any]:
    """Return a stable description of ``df``: row count + sorted-column hash."""
    cols = ",".join(sorted(df.columns.astype(str)))
    return {
        "source_path": str(source_path),
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns_hash": _sha256(cols),
    }


def config_hash(config_dict: dict[str, Any]) -> str:
    """SHA-256 of the canonical-JSON form of ``config_dict``."""
    payload = json.dumps(config_dict, sort_keys=True, default=str)
    return _sha256(payload)


def git_info(cwd: Path) -> tuple[str | None, bool]:
    """Return ``(commit_sha, dirty)``. Both ``None``/``False`` if git absent or call fails."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as e:
        _log.debug("git_info: subprocess failed: %s", e)
        return None, False

    if commit.returncode != 0:
        return None, False
    sha = commit.stdout.strip() or None
    dirty = bool(status.stdout.strip())
    return sha, dirty
