"""Dummy classifier factory."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier

from rsm_pipeline.models.schema import DummyModelConfig


def _make_dummy(cfg: DummyModelConfig, *, seed: int) -> DummyClassifier:
    kwargs: dict = {"strategy": cfg.strategy, "random_state": seed}
    if cfg.strategy == "constant":
        if cfg.constant is None:
            raise ValueError("dummy.strategy='constant' requires dummy.constant")
        kwargs["constant"] = cfg.constant
    return DummyClassifier(**kwargs)
