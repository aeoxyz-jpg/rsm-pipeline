"""``rsm-serve`` — FastAPI server over a TrainedBundle."""

from __future__ import annotations

import logging
import sys

import click
import joblib
import uvicorn

from rsm_pipeline.logging_setup import configure_logging
from rsm_pipeline.serving.api import build_app


@click.command(name="rsm-serve")
@click.option(
    "--model",
    "model_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to bundle.joblib produced by rsm-train.",
)
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Bind host. Use 0.0.0.0 to expose externally.",
)
@click.option(
    "--port",
    default=8000,
    type=int,
    show_default=True,
    help="Bind port.",
)
def main(model_path: str, host: str, port: int) -> None:
    configure_logging(level="INFO")
    log = logging.getLogger("rsm_pipeline.cli.serve")
    bundle = joblib.load(model_path)
    log.info(
        "rsm-serve: serving bundle=%s on http://%s:%d (build_meta=%s)",
        model_path,
        host,
        port,
        bundle.build_meta,
    )
    app = build_app(bundle)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
