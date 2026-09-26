"""Content-addressed disk cache for LLM requests (request hash → raw response).

Every logical LLM call is keyed by a sha256 of its *canonical request* (model,
endpoint, messages, token cap, schema, …) so that:
  * re-running any stage is free for already-answered requests (resumable);
  * batch and realtime results share one store (a smoke result is reused by the
    full batch);
  * changing the prompt/schema/model/seed re-keys automatically, and a manual
    ``cache_version`` bump invalidates everything intentionally.

On-disk layout: ``<cache_dir>/responses/<first2hex>/<full_sha256>.json`` (sharded
to keep directories small for ~10^5 files). Each record is self-describing
(request hash, the canonical request, the raw response text, usage, source).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path

log = logging.getLogger(__name__)


def request_hash(canonical: dict) -> str:
    """sha256 of a canonical request dict (sorted keys, unicode preserved)."""
    blob = json.dumps(canonical, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def cache_path(cache_dir: str | Path, h: str) -> Path:
    return Path(cache_dir) / "responses" / h[:2] / f"{h}.json"


def get(cache_dir: str | Path, h: str) -> dict | None:
    """Read a cache record, or ``None`` if absent **or unreadable**.

    A truncated or interleaved file is treated as a miss rather than raised: the callers
    (:func:`partition_cached`, the collect stages) then re-request it, which is the
    recoverable behaviour. Raising here aborts a whole run over one bad file.
    """
    p = cache_path(cache_dir, h)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        log.warning("corrupt cache record %s (%s); treating as a miss", p, exc)
        return None


def put(cache_dir: str | Path, h: str, record: dict) -> None:
    """Write a cache record atomically.

    Atomic because two worker threads can legitimately hold the *same* key: the key is a
    hash of the canonical request, so two distinct items with byte-identical prompts collide
    by design. A plain ``write_text`` lets one thread's truncate interleave with another's
    write, leaving valid JSON followed by the tail of the longer record. Write-then-
    ``os.replace`` is atomic on POSIX, so the loser of the race is simply overwritten by an
    equally valid file.
    """
    p = cache_path(cache_dir, h)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(json.dumps(record, ensure_ascii=False))
        os.replace(tmp, p)
    finally:
        tmp.unlink(missing_ok=True)


def partition_cached(cache_dir: str | Path, specs) -> tuple[dict, list]:
    """Split ``specs`` into ``(cached_records_by_custom_id, uncached_specs)``.

    ``specs`` are :class:`core.judging.requests.RequestSpec`. The returned cache
    records are keyed by ``spec.custom_id`` for easy reassembly downstream.
    """
    cached: dict[str, dict] = {}
    uncached = []
    for spec in specs:
        rec = get(cache_dir, spec.cache_key())
        if rec is not None:
            cached[spec.custom_id] = rec
        else:
            uncached.append(spec)
    return cached, uncached
