"""``rsm-predict`` — batch scoring CLI."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import joblib

from rsm_pipeline.logging_setup import configure_logging
from rsm_pipeline.serving.batch import score_batch
from rsm_pipeline.serving.io import _read_input, _write_output


@click.command(name="rsm-predict")
@click.option(
    "--model",
    "model_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to bundle.joblib produced by rsm-train.",
)
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Input CSV or Parquet file with raw feature columns.",
)
@click.option(
    "--output",
    "output_path",
    required=True,
    type=click.Path(),
    help="Output CSV or Parquet file (extension decides the format).",
)
@click.option(
    "--threshold",
    type=float,
    default=0.5,
    show_default=True,
    help="Probability threshold for the `predict` column.",
)
@click.option(
    "--include-score",
    is_flag=True,
    help="Also include the integer `score` column (requires scorer in bundle).",
)
def main(
    model_path: str,
    input_path: str,
    output_path: str,
    threshold: float,
    include_score: bool,
) -> None:
    configure_logging(level="INFO")
    log = logging.getLogger("rsm_pipeline.cli.predict")

    bundle = joblib.load(model_path)
    df = _read_input(Path(input_path))
    log.info(
        "rsm-predict: %d rows in (cols=%d) -> bundle=%s",
        len(df),
        len(df.columns),
        Path(model_path).name,
    )

    scored = score_batch(bundle, df, threshold=threshold, include_score=include_score)
    _write_output(scored, Path(output_path))

    extras = [c for c in scored.columns if c not in df.columns]
    log.info(
        "rsm-predict: wrote %d rows to %s (added cols: %s, threshold=%.2f, scored=%s)",
        len(scored),
        output_path,
        extras,
        threshold,
        "yes" if include_score else "no",
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
