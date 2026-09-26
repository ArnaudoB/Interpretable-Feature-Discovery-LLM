"""Exponential-backoff wrapper for realtime OpenAI calls and file uploads.

Batch *jobs* report per-line errors and don't need request-level retry; this is
for the synchronous calls (smoke tests, single scoring calls, file uploads) that
can transiently 429/5xx.
"""

from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)

_RETRYABLE = ("429", "500", "502", "503", "504", "timeout", "connection", "overloaded")


def with_backoff(fn, *args, tries: int = 5, base: float = 3.0, **kwargs):
    """Call ``fn(*args, **kwargs)`` retrying on transient errors with exp backoff."""
    last = None
    for attempt in range(tries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - we re-raise after exhausting retries
            last = exc
            msg = str(exc).lower()
            if attempt < tries - 1 and any(tok in msg for tok in _RETRYABLE):
                delay = base * (2**attempt)
                log.warning(
                    "retryable error (attempt %d/%d): %s — sleeping %.1fs",
                    attempt + 1,
                    tries,
                    str(exc)[:120],
                    delay,
                )
                time.sleep(delay)
                continue
            raise
    raise last  # pragma: no cover
