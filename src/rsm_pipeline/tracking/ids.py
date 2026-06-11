"""Run-ID generator: ``<utc-timestamp>-<6-hex-chars>``."""

from __future__ import annotations

import datetime as _dt
import secrets


def new_run_id() -> str:
    """Return a fresh run id, e.g. ``20260504T134210Z-a1b2c3``."""
    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}-{secrets.token_hex(3)}"
