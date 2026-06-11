"""Project-root discovery and path resolution.

All path handling in the package goes through these helpers — no string
concatenation, no ``os.path.join`` with hardcoded separators, no
absolute paths from the author's machine.
"""

from __future__ import annotations

import logging
from pathlib import Path

_log = logging.getLogger(__name__)


def project_root() -> Path:
    """Return the directory containing ``pyproject.toml``.

    Walks up from this file until ``pyproject.toml`` is found. Falls
    back to the current working directory (with a warning) if the
    package is being executed from a location that has no
    ``pyproject.toml`` ancestor.
    """
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    cwd = Path.cwd().resolve()
    _log.warning(
        "project_root: no pyproject.toml found above %s; falling back to cwd %s",
        here,
        cwd,
    )
    return cwd


def resolve_path(raw: str | Path, anchor: Path | None = None) -> Path:
    """Resolve ``raw`` to an absolute path.

    Absolute inputs are returned unchanged. Relative inputs are
    anchored on ``anchor`` (default: project root).
    """
    p = Path(raw)
    if p.is_absolute():
        return p
    return (anchor or project_root()) / p
