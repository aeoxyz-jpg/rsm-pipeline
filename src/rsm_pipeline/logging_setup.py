"""Logging configuration for the pipeline.

Idempotent: calling ``configure_logging`` twice clears existing handlers
before re-installing them. All FileHandlers use UTF-8 encoding.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)s %(name)s | %(message)s"


def configure_logging(level: str = "INFO", run_dir: Path | None = None) -> None:
    """Install stdout + optional file handlers on the root logger.

    Parameters
    ----------
    level : str
        Default level. Overridden by the ``RSM_LOG_LEVEL`` env var if set.
    run_dir : Path | None
        If given, a ``train.log`` file is written under this directory.
    """
    effective = os.environ.get("RSM_LOG_LEVEL", level).upper()
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(_FORMAT)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)

    root.setLevel(getattr(logging, effective, logging.INFO))

    # Quiet a few notoriously chatty libraries.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
