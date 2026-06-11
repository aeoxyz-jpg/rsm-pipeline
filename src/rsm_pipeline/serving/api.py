"""FastAPI app builder over a TrainedBundle."""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rsm_pipeline.serving.batch import score_batch


class PredictRequest(BaseModel):
    records: list[dict[str, Any]] = Field(min_length=1)
    threshold: float = 0.5
    include_score: bool = False


class PredictResponse(BaseModel):
    proba_1: list[float]
    predict: list[int]
    score: list[int] | None = None
    meta: dict[str, Any]


def build_app(bundle: Any) -> FastAPI:
    app = FastAPI(
        title="rsm-serve",
        version=str(bundle.build_meta.get("python", "unknown")),
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/version")
    def version() -> dict[str, Any]:
        return {
            "build_meta": bundle.build_meta,
            "raw_input_columns": list(bundle.raw_input_columns),
            "feats_after_fs": list(bundle.feats_after_fs),
            "has_scorer": bundle.scorer is not None,
            "target": bundle.target,
        }

    def _do_predict(req: PredictRequest) -> PredictResponse:
        df = pd.DataFrame(req.records)
        try:
            scored = score_batch(
                bundle,
                df,
                threshold=req.threshold,
                include_score=req.include_score,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return PredictResponse(
            proba_1=scored["proba_1"].tolist(),
            predict=scored["predict"].tolist(),
            score=(scored["score"].tolist() if req.include_score else None),
            meta={"n_in": len(df), "n_out": len(scored)},
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        return _do_predict(req)

    @app.post("/predict_batch", response_model=PredictResponse)
    def predict_batch(req: PredictRequest) -> PredictResponse:
        return _do_predict(req)

    return app
