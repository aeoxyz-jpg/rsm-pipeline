"""YAML -> ``RsmConfig`` loader with project-root path resolution."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from rsm_pipeline.config.schema import RsmConfig
from rsm_pipeline.paths import project_root, resolve_path

_log = logging.getLogger(__name__)


def load_config(path: str | Path) -> RsmConfig:
    """Read ``path`` as YAML and validate as :class:`RsmConfig`.

    Path fields inside the config (``data.source.path`` and
    ``run.output_root``) are resolved against the project root and
    written back as absolute strings. The returned model is frozen.

    Parameters
    ----------
    path:
        Path to the YAML config file.

    Returns
    -------
    RsmConfig
        Validated, frozen config object with all paths resolved.

    Examples
    --------
    >>> cfg = load_config("configs/example_config.yaml")
    >>> cfg.run.name
    'my_experiment'
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = RsmConfig.model_validate(raw)

    root = project_root()
    resolved = cfg.model_dump(mode="json")
    resolved["data"]["source"]["path"] = str(
        resolve_path(resolved["data"]["source"]["path"], root)
    )
    resolved["run"]["output_root"] = str(
        resolve_path(resolved["run"]["output_root"], root)
    )
    cfg = RsmConfig.model_validate(resolved)

    _log.info(
        "loaded config from %s (run=%s seed=%d)",
        p,
        cfg.run.name,
        cfg.run.seed,
    )
    return cfg
