"""Tiny shared base for Pydantic models in the package.

Lives at the package root with no other imports so any sub-module can
``from rsm_pipeline._frozen import _Frozen`` without creating cycles.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
