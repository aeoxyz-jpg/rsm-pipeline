"""Random over/under sampler factories."""

from __future__ import annotations

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from rsm_pipeline.imbalance.schema import RandomOverConfig, RandomUnderConfig


def _make_random_oversampler(cfg: RandomOverConfig, *, random_state: int):
    return RandomOverSampler(
        sampling_strategy=cfg.sampling_strategy,
        random_state=random_state,
    )


def _make_random_undersampler(cfg: RandomUnderConfig, *, random_state: int):
    return RandomUnderSampler(
        sampling_strategy=cfg.sampling_strategy,
        random_state=random_state,
    )
