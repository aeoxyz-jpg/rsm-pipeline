"""``rsm-monitor`` — PSI / CSI / data-quality / performance drift CLI."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click
import joblib

from rsm_pipeline.logging_setup import configure_logging
from rsm_pipeline.monitoring.apply import apply_monitoring
from rsm_pipeline.serving.io import _read_input


@click.command(name="rsm-monitor")
@click.option(
    "--bundle",
    "bundle_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--reference",
    "reference_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--current",
    "current_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option("--output", "output_dir", required=True, type=click.Path())
@click.option(
    "--reference-metrics",
    "ref_metrics_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option("--has-labels", is_flag=True)
@click.option(
    "--target", default=None, help="Target column; defaults to bundle.target."
)
@click.option("--n-bins", default=10, type=int, show_default=True)
@click.option("--minor-threshold", default=0.1, type=float, show_default=True)
@click.option("--major-threshold", default=0.25, type=float, show_default=True)
def main(
    bundle_path: str,
    reference_path: str,
    current_path: str,
    output_dir: str,
    ref_metrics_path: Optional[str],
    has_labels: bool,
    target: Optional[str],
    n_bins: int,
    minor_threshold: float,
    major_threshold: float,
) -> None:
    configure_logging(level="INFO")
    log = logging.getLogger("rsm_pipeline.cli.monitor")

    bundle = joblib.load(bundle_path)
    reference = _read_input(Path(reference_path))
    current = _read_input(Path(current_path))
    ref_metrics = None
    if ref_metrics_path:
        ref_metrics = json.loads(Path(ref_metrics_path).read_text(encoding="utf-8"))

    log.info(
        "rsm-monitor: bundle=%s ref=%dx%d cur=%dx%d has_labels=%s",
        Path(bundle_path).name,
        *reference.shape,
        *current.shape,
        has_labels,
    )

    summary = apply_monitoring(
        bundle,
        reference,
        current,
        Path(output_dir),
        reference_metrics=ref_metrics,
        has_labels=has_labels,
        target=target,
        n_bins=n_bins,
        minor_threshold=minor_threshold,
        major_threshold=major_threshold,
    )
    log.info(
        "score_psi=%.4f tier=%s | csi: max=%.4f tier=%s n_major=%d",
        summary["score_psi"]["value"],
        summary["score_psi"]["tier"],
        summary["csi"]["max_value"] or 0.0,
        summary["csi"]["tier"],
        summary["csi"]["n_major"],
    )
    if summary.get("performance"):
        log.info(
            "performance: tier=%s delta_roc_auc=%s",
            summary["performance"]["tier"],
            summary["performance"]["delta_roc_auc"],
        )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
