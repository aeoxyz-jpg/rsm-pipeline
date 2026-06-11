"""Search-space translators: dialect -> sklearn / optuna."""

from __future__ import annotations

from typing import Any

from scipy.stats import loguniform, randint, uniform

from rsm_pipeline.tuning.schema import (
    CategoricalDist,
    FloatDist,
    IntDist,
    SearchSpaceValue,
)


def _to_sklearn_grid(space: dict[str, SearchSpaceValue]) -> dict[str, list]:
    out: dict[str, list] = {}
    for k, v in space.items():
        if not isinstance(v, list):
            raise ValueError(
                f"grid requires list-valued search_space[{k!r}] (got {type(v).__name__})"
            )
        out[k] = v
    return out


def _to_sklearn_distributions(
    space: dict[str, SearchSpaceValue],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in space.items():
        if isinstance(v, list):
            out[k] = v
            continue
        if isinstance(v, FloatDist):
            if v.log:
                out[k] = loguniform(v.low, v.high)
            else:
                out[k] = uniform(loc=v.low, scale=v.high - v.low)
        elif isinstance(v, IntDist):
            out[k] = randint(v.low, v.high + 1)
        elif isinstance(v, CategoricalDist):
            out[k] = list(v.choices)
        else:
            raise TypeError(f"unsupported space dist for {k!r}: {type(v).__name__}")
    return out


def _apply_to_optuna_trial(space: dict[str, SearchSpaceValue], trial) -> dict:
    """Build {param: value} by calling trial.suggest_*."""
    params: dict[str, Any] = {}
    for k, v in space.items():
        if isinstance(v, list):
            params[k] = trial.suggest_categorical(k, v)
        elif isinstance(v, FloatDist):
            if v.step is not None and not v.log:
                params[k] = trial.suggest_float(k, v.low, v.high, step=v.step)
            else:
                params[k] = trial.suggest_float(k, v.low, v.high, log=v.log)
        elif isinstance(v, IntDist):
            params[k] = trial.suggest_int(k, v.low, v.high, step=v.step, log=v.log)
        elif isinstance(v, CategoricalDist):
            params[k] = trial.suggest_categorical(k, v.choices)
        else:
            raise TypeError(f"unsupported optuna spec for {k!r}: {type(v).__name__}")
    return params
