"""Stage mechanics shared by this experiment's LLM steps (02, 03, 05, 05b, 06, 07, 08).

Every LLM step builds its request specs deterministically from shipped inputs and the prompts
in ``prompts/``, then runs one of three stages:

* ``smoke``   -- a handful of specs realtime, to measure real token cost. Only the specs the
  cache does NOT hold are called; a cached record is reused.
* ``submit``  -- every uncached spec to the OpenAI Batch API (distinct requests only: two
  items with byte-identical prompts share one cache key, so one call answers both).
* ``collect`` -- poll whatever ``submit`` recorded, then read every spec from the cache and
  assemble. With the shipped caches nothing is pending, so this is a pure cache read that
  needs no key; it RAISES if any spec misses (``--allow-missing`` assembles without them).
  A cached response that does not parse is not a miss: it is replayed, and the step's
  assembly drops it exactly as the original run did.

A few steps add their own stages -- ``run`` (a realtime call where the run made one, e.g.
the taxonomy and the elicitations), ``rederive`` (offline, from a saved parse, where the
response itself was not cached), ``retry`` / ``embed``. Every stage that can spend money
(``PAID``) refuses to run without ``--yes``.

Cache keys: ``RequestSpec.cache_key()`` is sha256 of the canonical request (endpoint, model,
system, user, max_output_tokens, temperature, seed, json_schema, reasoning_effort,
cache_version; ``thinking``/``thinking_level`` only when set). ``core.judging.requests`` builds
the same request dict the runs sent for these OpenAI Responses/chat calls (the optional keys of
other providers are never set by a CMV spec), so a spec rebuilt here hashes to the key its
response was filed under.
``tests/test_cmv_replay.py`` checks that for every shipped cache.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import _paths  # noqa: F401
from core.judging import cache
from core.judging.batch import (
    build_jsonl_chunked,
    collect_into_cache,
    poll_until_done,
    realtime_call,
    submit_batches,
)
from core.judging.cache import partition_cached
from core.judging.requests import RESPONSES_ENDPOINT

log = logging.getLogger(__name__)

PAID = ("smoke", "submit", "run", "retry", "embed")


def add_stage_args(ap, stages=("smoke", "submit", "collect")):
    """The arguments every LLM step shares."""
    ap.add_argument("stage", choices=stages)
    ap.add_argument(
        "--out-dir",
        default=None,
        help="mirror the cell tree under DIR (a replay to diff against the shipped "
        "cell); upstream inputs are read from DIR when regenerated there",
    )
    ap.add_argument(
        "--in-place", action="store_true", help="write into the shipped cell (a genuine re-run)"
    )
    ap.add_argument(
        "--yes",
        action="store_true",
        help="required by the stages that can issue paid API calls (smoke, submit, "
        "run, retry, embed); `collect` is a pure cache read when every spec "
        "is present",
    )
    ap.add_argument(
        "--allow-missing",
        action="store_true",
        help="collect: assemble from whatever is cached instead of raising on misses",
    )
    return ap


def guard(stage: str, yes: bool) -> None:
    if stage in PAID and not yes:
        raise SystemExit(
            f"`{stage}` can issue paid API calls -- pass --yes to confirm. "
            f"Everything the paper reports replays from the shipped caches with "
            f"`collect`, which needs no key and spends nothing."
        )


def client():
    """An OpenAI client, created only when a stage actually needs the API."""
    from core.openai_client import get_client

    try:
        return get_client()
    except KeyError:
        raise SystemExit(
            "OPENAI_API_KEY is not set (nor in .env); this stage needs the API. "
            "`collect` replays the shipped cache without one."
        ) from None


def replay(specs, cache_dir, *, allow_missing=False, label="") -> dict:
    """Every spec's cached record, keyed by ``custom_id``; raises on a miss.

    A spec whose record carries an ``error`` counts as a miss.
    """
    got, missing = {}, []
    for s in specs:
        r = cache.get(cache_dir, s.cache_key())
        if r is None or r.get("error"):
            missing.append(s)
        else:
            got[s.custom_id] = r
    n_keys = len({s.cache_key() for s in specs})
    miss = [s.custom_id for s in missing]
    print(
        f"[replay{(' ' + label) if label else ''}] {len(specs)} specs ({n_keys} distinct "
        f"requests): {len(got)} cached, {len(missing)} missing"
    )
    if miss and not allow_missing:
        raise SystemExit(
            f"{len(miss)} spec(s) miss the cache {cache_dir}: "
            f"{miss[:5]}{' ...' if len(miss) > 5 else ''}. A request "
            f"drifted from the one the run sent, or the run never collected it. "
            f"Run the step's paid stage (`submit`/`run` --yes) to request them, or "
            f"--allow-missing to assemble without them."
        )
    return got


def smoke(specs, cache_dir, cl=None) -> list[dict]:
    """Records for ``specs``: cached ones reused, the rest called realtime and cached."""
    out = []
    for s in specs:
        rec = cache.get(cache_dir, s.cache_key())
        if rec is None or rec.get("error"):
            cl = cl or client()
            rec = realtime_call(cl, s)
            cache.put(cache_dir, s.cache_key(), rec)
        out.append(rec)
    return out


def submit(specs, cache_dir, batch_dir: Path, endpoint: str = RESPONSES_ENDPOINT) -> None:
    """Submit every uncached, distinct request as one or more 24h batches."""
    cached, todo = partition_cached(cache_dir, specs)
    batch_dir.mkdir(parents=True, exist_ok=True)
    uniq = list({s.cache_key(): s for s in todo}.values())
    print(
        f"[submit] {len(specs)} specs | cached {len(cached)} | to submit {len(uniq)} "
        f"distinct requests"
    )
    if not uniq:
        (batch_dir / "meta.json").write_text(json.dumps({"batch_ids": [], "status": "all_cached"}))
        print("[submit] all cached -- go straight to `collect`.")
        return
    paths = build_jsonl_chunked(uniq, batch_dir)
    submit_batches(client(), paths, endpoint, batch_dir / "meta.json")
    print(f"[submit] {len(paths)} batch(es) submitted -- run `collect` when they resolve (24h).")


def poll(specs, cache_dir, batch_dir: Path) -> None:
    """Collect any batches ``submit`` recorded under ``batch_dir`` into the cache.

    Accepts both meta shapes a run may have written (``batch_ids`` list, or one ``batch_id``).
    A no-op -- and no client is created -- when nothing was submitted.
    """
    meta_p = batch_dir / "meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    ids = meta.get("batch_ids") or ([meta["batch_id"]] if meta.get("batch_id") else [])
    if not ids:
        return
    cl = client()
    by_cid = {s.custom_id: s for s in specs}
    for i, bid in enumerate(ids):
        print(f"[collect] batch {i + 1}/{len(ids)}: {bid}")
        b = poll_until_done(cl, bid, interval_s=60)
        collect_into_cache(cl, b, by_cid, cache_dir, batch_dir / f"chunk{i:02d}")
