"""Batch lifecycles (OpenAI, Anthropic, Gemini) + realtime calls, composed with the cache.

A logical phase resolves like this (idempotent / resumable at every stage):
  build   → assemble RequestSpecs, write a sidecar manifest
  submit  → partition_cached → JSONL of only-uncached specs → upload → batches.create
  collect → poll terminal status → download output/error JSONL → write each result
            into the content-addressed cache
  analyze → reassemble ALL specs from cache (cached + freshly collected) by hash

``run_smoke`` fires the first N specs against the realtime endpoint with the SAME
body shape as the batch lines, and writes them to the cache too — so a smoke result
is reused by the full batch and never re-billed.

The orchestration is independent of any one schema: the body comes from
``RequestSpec.body()`` and the parser is the uniform extractor below. Realtime calls
dispatch on ``spec.endpoint`` to the OpenAI (Responses / chat), Anthropic (Messages),
Gemini (generateContent) and DeepSeek (OpenAI-dialect chat at its own base_url) clients.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import time
from pathlib import Path

from core.judging import cache
from core.judging.requests import (
    CHAT_ENDPOINT,
    DEEPSEEK_ENDPOINT,
    GENERATE_ENDPOINT,
    MESSAGES_ENDPOINT,
    RESPONSES_ENDPOINT,
)
from core.judging.retry import with_backoff

log = logging.getLogger(__name__)

_TERMINAL = {"completed", "failed", "expired", "cancelled", "canceling"}


# --------------------------------------------------------------------------- #
# Response extraction (realtime SDK objects and batch-output dicts)
# --------------------------------------------------------------------------- #
def _usage_dict(usage, endpoint: str) -> dict:
    if usage is None:
        return {}
    g = (lambda k: getattr(usage, k, None)) if not isinstance(usage, dict) else usage.get
    if endpoint == GENERATE_ENDPOINT:
        # thoughts_token_count is billed as output but is NOT included in
        # candidates_token_count -- omitting it undercounts spend on thinking-capable models.
        def _gv(*names):
            for n in names:
                v = g(n)
                if v:
                    return v
            return 0

        return {
            "input_tokens": _gv("prompt_token_count", "promptTokenCount"),
            "output_tokens": (
                _gv("candidates_token_count", "candidatesTokenCount")
                + _gv("thoughts_token_count", "thoughtsTokenCount")
            ),
            "thoughts_tokens": _gv("thoughts_token_count", "thoughtsTokenCount"),
        }
    if endpoint in (RESPONSES_ENDPOINT, MESSAGES_ENDPOINT):
        return {"input_tokens": g("input_tokens") or 0, "output_tokens": g("output_tokens") or 0}
    if endpoint == DEEPSEEK_ENDPOINT:
        # DeepSeek splits input into cache hit/miss and prices them ~31x apart, so record
        # both; `reasoning_tokens` is captured to confirm that thinking is disabled.
        out = {
            "input_tokens": g("prompt_tokens") or 0,
            "output_tokens": g("completion_tokens") or 0,
        }
        for src, dst in (
            ("prompt_cache_hit_tokens", "cache_hit_tokens"),
            ("prompt_cache_miss_tokens", "cache_miss_tokens"),
            ("reasoning_tokens", "reasoning_tokens"),
        ):
            v = g(src)
            if v is not None:
                out[dst] = v
        return out
    return {"input_tokens": g("prompt_tokens") or 0, "output_tokens": g("completion_tokens") or 0}


def _text_from_chat(body: dict) -> str:
    try:
        return body["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


def _text_from_responses(body: dict) -> str:
    if isinstance(body.get("output_text"), str):
        return body["output_text"]
    parts = []
    for item in body.get("output") or []:
        for c in item.get("content") or []:
            if c.get("type") in ("output_text", "text") and c.get("text"):
                parts.append(c["text"])
    return "".join(parts)


def _text_from_messages(resp) -> str:
    """JSON string from the forced ``tool_use`` block of an Anthropic response.

    Re-serializing the tool input keeps the cache-record shape identical to the OpenAI
    paths, so the cell-local ``parse_comparative`` and the collect stage are unchanged.
    Falls back to concatenated ``text`` blocks if the model answered in prose instead
    (``parse_comparative`` already tolerates fenced/loose JSON).
    """
    parts = []
    for block in getattr(resp, "content", None) or []:
        btype = getattr(block, "type", None)
        if btype == "tool_use":
            return json.dumps(getattr(block, "input", None), ensure_ascii=False)
        if btype == "text" and getattr(block, "text", None):
            parts.append(block.text)
    return "".join(parts)


def _record(spec, text: str, usage: dict, source: str, batch_id=None, error=None) -> dict:
    return {
        "request_hash": spec.cache_key(),
        "custom_id": spec.custom_id,
        "model": spec.model,
        "endpoint": spec.endpoint,
        "text": text,
        "usage": usage,
        "source": source,
        "batch_id": batch_id,
        "error": error,
    }


# --------------------------------------------------------------------------- #
# Realtime (smoke / single calls)
# --------------------------------------------------------------------------- #
def realtime_call(client, spec) -> dict:
    """One synchronous call for ``spec``; returns a cache record (with backoff)."""
    body = spec.body()
    if spec.endpoint == CHAT_ENDPOINT:
        resp = with_backoff(client.chat.completions.create, **body)
        text = resp.choices[0].message.content or ""
        usage = _usage_dict(getattr(resp, "usage", None), spec.endpoint)
    elif spec.endpoint == RESPONSES_ENDPOINT:
        resp = with_backoff(client.responses.create, **body)
        text = getattr(resp, "output_text", "") or ""
        usage = _usage_dict(getattr(resp, "usage", None), spec.endpoint)
    elif spec.endpoint == DEEPSEEK_ENDPOINT:
        # `thinking` is not a typed parameter of the OpenAI SDK, so it rides in extra_body.
        b = dict(body)
        extra = {k: b.pop(k) for k in ("thinking",) if k in b}
        resp = with_backoff(client.chat.completions.create, **b, extra_body=extra)
        msg = resp.choices[0].message
        calls = getattr(msg, "tool_calls", None) or []
        # The forced tool call's arguments are already a JSON string, so storing them
        # verbatim keeps the cache-record shape identical to every other provider.
        text = calls[0].function.arguments if calls else (msg.content or "")
        usage = _usage_dict(getattr(resp, "usage", None), spec.endpoint)
    elif spec.endpoint == GENERATE_ENDPOINT:
        cfg = dict(body["generationConfig"])
        cfg["systemInstruction"] = body["systemInstruction"]
        resp = with_backoff(
            client.models.generate_content,
            model=body["model"],
            contents=body["contents"],
            config=cfg,
        )
        text = getattr(resp, "text", "") or ""
        usage = _usage_dict(getattr(resp, "usage_metadata", None), spec.endpoint)
    elif spec.endpoint == MESSAGES_ENDPOINT:
        resp = with_backoff(client.messages.create, **body)
        text = _text_from_messages(resp)
        usage = _usage_dict(getattr(resp, "usage", None), spec.endpoint)
    else:
        raise ValueError(f"unsupported endpoint {spec.endpoint!r}")
    return _record(spec, text, usage, source="realtime")


def run_smoke(client, specs, cache_dir) -> list[dict]:
    """Run ``specs`` realtime, writing each to the cache; returns the records."""
    out = []
    for spec in specs:
        rec = realtime_call(client, spec)
        cache.put(cache_dir, spec.cache_key(), rec)
        out.append(rec)
    return out


def run_concurrent(client, specs, cache_dir, max_workers: int = 8, progress_every: int = 100):
    """Run ``specs`` realtime across a thread pool; cache successes; return (records, errors).

    Each call already retries transient errors (``realtime_call`` → ``with_backoff``).
    Hard failures are recorded but NOT cached, so a re-run retries only those. The
    OpenAI SDK client is safe to share across threads.
    """
    recs, errors = [], []
    done = 0

    def work(spec):
        try:
            rec = realtime_call(client, spec)
            cache.put(cache_dir, spec.cache_key(), rec)  # cache only on success
            return rec
        except Exception as exc:  # noqa: BLE001
            return _record(spec, "", {}, "realtime", error=str(exc)[:300])

    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(work, s): s for s in specs}
        for fut in cf.as_completed(futures):
            rec = fut.result()
            recs.append(rec)
            if rec.get("error"):
                errors.append((futures[fut].custom_id, rec["error"]))
            done += 1
            if done % progress_every == 0:
                log.info("  sync %d/%d (%d errors)", done, len(specs), len(errors))
    return recs, errors


# --------------------------------------------------------------------------- #
# Batch lifecycle
# --------------------------------------------------------------------------- #
def build_jsonl(specs, out_path: str | Path) -> int:
    """Write one batch request line per spec; returns the count written."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for spec in specs:
            f.write(
                json.dumps(
                    {
                        "custom_id": spec.custom_id,
                        "method": "POST",
                        "url": spec.endpoint,
                        "body": spec.body(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return len(specs)


#: OpenAI rejects a batch input file over 200 MB. Stay well under it -- a 16-criterion
#: comparative call is ~12 KB of JSON, so a full-scale cell blows the cap in one file
#: (the 54,576-comparison full-budget run needed 4 chunks).
MAX_BATCH_BYTES = 150_000_000


def build_jsonl_chunked(specs, out_dir: str | Path, max_bytes: int = MAX_BATCH_BYTES) -> list:
    """Write batch requests split across as many files as the size cap requires.

    Returns the list of written paths. Splitting is by serialized byte length, not request
    count, because per-request size varies with the exchange texts.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("requests_*.jsonl"):
        stale.unlink()

    paths, fh, size, idx = [], None, 0, 0
    try:
        for spec in specs:
            line = (
                json.dumps(
                    {
                        "custom_id": spec.custom_id,
                        "method": "POST",
                        "url": spec.endpoint,
                        "body": spec.body(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            b = len(line.encode("utf-8"))
            if fh is None or size + b > max_bytes:
                if fh is not None:
                    fh.close()
                p = out_dir / f"requests_{idx:02d}.jsonl"
                paths.append(p)
                fh = p.open("w")
                size, idx = 0, idx + 1
            fh.write(line)
            size += b
    finally:
        if fh is not None:
            fh.close()
    log.info("wrote %d batch chunk(s) to %s", len(paths), out_dir)
    return paths


def submit_batches(
    client, jsonl_paths, endpoint: str, meta_path: str | Path, completion_window: str = "24h"
) -> dict:
    """Submit several chunk files as separate batches; record every id in one meta file."""
    meta_path = Path(meta_path)
    entries = []
    for p in jsonl_paths:
        up = with_backoff(lambda: client.files.create(file=Path(p).open("rb"), purpose="batch"))
        batch = with_backoff(
            lambda: client.batches.create(
                input_file_id=up.id, endpoint=endpoint, completion_window=completion_window
            )
        )
        entries.append(
            {
                "batch_id": batch.id,
                "input_file_id": up.id,
                "file": str(Path(p).name),
                "status": batch.status,
            }
        )
        log.info("submitted batch %s from %s (%s)", batch.id, Path(p).name, batch.status)
    meta = {
        "batch_ids": [e["batch_id"] for e in entries],
        "chunks": entries,
        "endpoint": endpoint,
        "completion_window": completion_window,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def submit_batch(
    client,
    jsonl_path: str | Path,
    endpoint: str,
    meta_path: str | Path,
    completion_window: str = "24h",
) -> dict:
    jsonl_path, meta_path = Path(jsonl_path), Path(meta_path)
    up = with_backoff(lambda: client.files.create(file=jsonl_path.open("rb"), purpose="batch"))
    batch = with_backoff(
        lambda: client.batches.create(
            input_file_id=up.id, endpoint=endpoint, completion_window=completion_window
        )
    )
    meta = {
        "batch_id": batch.id,
        "input_file_id": up.id,
        "endpoint": endpoint,
        "status": batch.status,
        "completion_window": completion_window,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info(
        "submitted batch %s (%s requests, %s)",
        batch.id,
        getattr(batch, "request_counts", "?"),
        endpoint,
    )
    return meta


def poll_until_done(
    client,
    batch_id: str,
    interval_s: int = 60,
    max_wait_s: int = 86400,
    meta_path: str | Path | None = None,
):
    interval_s = max(60, int(interval_s))
    waited, last = 0, None
    while True:
        batch = with_backoff(client.batches.retrieve, batch_id)
        if batch.status != last:
            log.info(
                "batch %s status=%s counts=%s",
                batch_id,
                batch.status,
                getattr(batch, "request_counts", None),
            )
            last = batch.status
            if meta_path:
                Path(meta_path).write_text(
                    json.dumps({"batch_id": batch_id, "status": batch.status}, indent=2)
                )
        if batch.status in _TERMINAL:
            return batch
        if waited >= max_wait_s:
            raise TimeoutError(
                f"batch {batch_id} not done after {max_wait_s}s (status={batch.status})"
            )
        time.sleep(interval_s)
        waited += interval_s


def collect_into_cache(client, batch, specs_by_custom_id: dict, cache_dir, batch_dir) -> dict:
    """Download outputs/errors, write each result to the cache. Returns a summary."""
    batch_dir = Path(batch_dir)
    batch_dir.mkdir(parents=True, exist_ok=True)
    n_ok = n_err = 0

    if getattr(batch, "output_file_id", None):
        content = with_backoff(client.files.content, batch.output_file_id).text
        (batch_dir / "batch_output.jsonl").write_text(content)
        for line in content.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            cid = row.get("custom_id")
            spec = specs_by_custom_id.get(cid)
            if spec is None:
                continue
            body = (row.get("response") or {}).get("body") or {}
            text = (
                _text_from_chat(body)
                if spec.endpoint == CHAT_ENDPOINT
                else _text_from_responses(body)
            )
            usage = _usage_dict(body.get("usage"), spec.endpoint)
            cache.put(cache_dir, spec.cache_key(), _record(spec, text, usage, "batch", batch.id))
            n_ok += 1

    if getattr(batch, "error_file_id", None):
        econtent = with_backoff(client.files.content, batch.error_file_id).text
        (batch_dir / "batch_errors.jsonl").write_text(econtent)
        n_err = sum(1 for ln in econtent.splitlines() if ln.strip())

    summary = {"ok": n_ok, "errors": n_err, "batch_id": batch.id, "status": batch.status}
    (batch_dir / "collect_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("collected batch %s: %d ok, %d errors", batch.id, n_ok, n_err)
    return summary


# --------------------------------------------------------------------------- #
# Anthropic Message Batches lifecycle (50% of realtime price)
# --------------------------------------------------------------------------- #
#: Anthropic caps a batch at 100k requests / 256 MB. A comparative call is ~11 KB,
#: so a 2,000-pair cell is ~22 MB and fits in a single batch.
ANTHROPIC_MAX_REQUESTS = 100_000

#: ``batch.processing_status`` values that mean "no longer running".
_ANTHROPIC_TERMINAL = {"ended", "canceled", "cancelled", "expired"}


def anthropic_submit(client, specs, meta_path: str | Path) -> dict:
    """Create one Message Batch from ``specs``; record the id in ``meta_path``.

    Anthropic takes the requests inline (``{custom_id, params}``) rather than via a
    file upload, so there is no JSONL/Files step as on the OpenAI path.
    """
    if len(specs) > ANTHROPIC_MAX_REQUESTS:
        raise ValueError(f"{len(specs)} requests exceeds the {ANTHROPIC_MAX_REQUESTS} cap")
    requests = [{"custom_id": s.custom_id, "params": s.body()} for s in specs]
    batch = with_backoff(lambda: client.messages.batches.create(requests=requests))
    meta = {
        "batch_id": batch.id,
        "n_requests": len(specs),
        "processing_status": batch.processing_status,
        "provider": "anthropic",
    }
    meta_path = Path(meta_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("submitted anthropic batch %s (%d requests)", batch.id, len(specs))
    return meta


def anthropic_poll_until_done(
    client,
    batch_id: str,
    interval_s: int = 30,
    max_wait_s: int = 86400,
    meta_path: str | Path | None = None,
):
    """Poll until ``processing_status`` leaves the running state."""
    interval_s = max(10, int(interval_s))
    waited, last = 0, None
    while True:
        batch = with_backoff(client.messages.batches.retrieve, batch_id)
        status = batch.processing_status
        counts = getattr(batch, "request_counts", None)
        if status != last:
            log.info("batch %s status=%s counts=%s", batch_id, status, counts)
            last = status
            if meta_path:
                Path(meta_path).write_text(
                    json.dumps(
                        {
                            "batch_id": batch_id,
                            "processing_status": status,
                            "request_counts": str(counts),
                            "provider": "anthropic",
                        },
                        indent=2,
                    )
                )
        if status in _ANTHROPIC_TERMINAL:
            return batch
        if waited >= max_wait_s:
            raise TimeoutError(f"batch {batch_id} not done after {max_wait_s}s (status={status})")
        time.sleep(interval_s)
        waited += interval_s


def anthropic_collect_into_cache(
    client, batch_id: str, specs_by_custom_id: dict, cache_dir, batch_dir
) -> dict:
    """Stream results into the content-addressed cache. Only successes are cached.

    Results arrive in arbitrary order, so every entry is keyed by ``custom_id`` back to
    its spec — never by position. Non-succeeded entries are left uncached so a re-run
    retries exactly those, for free.
    """
    batch_dir = Path(batch_dir)
    batch_dir.mkdir(parents=True, exist_ok=True)
    n_ok = n_err = n_unknown = 0
    errors: list[dict] = []

    for res in with_backoff(client.messages.batches.results, batch_id):
        cid = res.custom_id
        spec = specs_by_custom_id.get(cid)
        if spec is None:
            n_unknown += 1
            continue
        rtype = res.result.type
        if rtype != "succeeded":
            n_err += 1
            errors.append(
                {
                    "custom_id": cid,
                    "type": rtype,
                    "error": str(getattr(res.result, "error", ""))[:300],
                }
            )
            continue
        msg = res.result.message
        text = _text_from_messages(msg)
        usage = _usage_dict(getattr(msg, "usage", None), MESSAGES_ENDPOINT)
        cache.put(cache_dir, spec.cache_key(), _record(spec, text, usage, "batch", batch_id))
        n_ok += 1

    summary = {
        "ok": n_ok,
        "errors": n_err,
        "unknown_custom_ids": n_unknown,
        "batch_id": batch_id,
        "provider": "anthropic",
    }
    (batch_dir / "collect_summary.json").write_text(json.dumps(summary, indent=2))
    if errors:
        (batch_dir / "batch_errors.json").write_text(json.dumps(errors, indent=2))
    log.info("collected anthropic batch %s: %d ok, %d errors", batch_id, n_ok, n_err)
    return summary


# --------------------------------------------------------------------------- #
# Gemini Batch API lifecycle (50% of realtime price)
# --------------------------------------------------------------------------- #
#: Inlined requests are capped at ~20 MB total. A 2,000-pair comparative cell serializes
#: to ~19.5 MB, which is too close, so this path always goes through the Files API.
_GEMINI_TERMINAL = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
}


def _gemini_state(job) -> str:
    st = getattr(job, "state", None)
    return getattr(st, "name", None) or str(st)


def gemini_build_jsonl(specs, out_path: str | Path) -> int:
    """Write the batch input JSONL: one ``{"key", "request"}`` line per spec."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for spec in specs:
            body = spec.body()
            f.write(
                json.dumps(
                    {
                        "key": spec.custom_id,
                        "request": {k: v for k, v in body.items() if k != "model"},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return len(specs)


def gemini_submit(
    client, specs, jsonl_path, meta_path: str | Path, display_name: str | None = None
) -> dict:
    """Upload the JSONL and start a batch job; record its name in ``meta_path``.

    ``batches.create`` is NOT idempotent: a 429 can come back after the job was already
    created server-side, so blindly retrying risks a second billable job for the same
    work. Instead the job is created in a single attempt under a deterministic
    ``display_name``, and on failure we look for a job already carrying that name and
    adopt it rather than creating another. The upload (which is free and safe to retry)
    is persisted to ``meta_path`` first, so a re-run reuses it instead of re-sending
    ~20 MB.
    """
    meta_path = Path(meta_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    prior = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    display_name = display_name or f"interp-scoring-{Path(jsonl_path).stem}"

    src_name = prior.get("input_file")
    if src_name:
        log.info("reusing uploaded batch input %s", src_name)
        n = prior.get("n_requests", len(specs))
    else:
        n = gemini_build_jsonl(specs, jsonl_path)
        up = with_backoff(
            lambda: client.files.upload(
                file=str(jsonl_path),
                config={"mime_type": "application/jsonl", "display_name": Path(jsonl_path).name},
            )
        )
        src_name = up.name
        meta_path.write_text(
            json.dumps({"input_file": src_name, "n_requests": n, "provider": "gemini"}, indent=2)
        )

    def _find_existing():
        try:
            for j in client.batches.list(config={"page_size": 50}):
                if j.display_name == display_name:
                    return j
        except Exception:  # noqa: BLE001 - listing is best-effort
            pass
        return None

    try:
        job = client.batches.create(
            model=specs[0].model, src=src_name, config={"display_name": display_name}
        )
    except Exception:
        job = _find_existing()
        if job is None:
            raise
        log.warning(
            "create failed but job %s already exists under %r -- adopting it, "
            "not creating a second one",
            job.name,
            display_name,
        )

    meta = {
        "batch_name": job.name,
        "input_file": src_name,
        "n_requests": n,
        "display_name": display_name,
        "state": _gemini_state(job),
        "provider": "gemini",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("submitted gemini batch %s (%d requests, src=%s)", job.name, n, src_name)
    return meta


def gemini_poll_until_done(
    client,
    batch_name: str,
    interval_s: int = 30,
    max_wait_s: int = 86400,
    meta_path: str | Path | None = None,
):
    interval_s = max(10, int(interval_s))
    waited, last = 0, None
    while True:
        job = with_backoff(client.batches.get, name=batch_name)
        state = _gemini_state(job)
        if state != last:
            log.info("batch %s state=%s", batch_name, state)
            last = state
            if meta_path:
                Path(meta_path).write_text(
                    json.dumps(
                        {"batch_name": batch_name, "state": state, "provider": "gemini"}, indent=2
                    )
                )
        if state in _GEMINI_TERMINAL:
            return job
        if waited >= max_wait_s:
            raise TimeoutError(f"batch {batch_name} not done after {max_wait_s}s (state={state})")
        time.sleep(interval_s)
        waited += interval_s


def _gemini_text_from_response(resp: dict) -> str:
    parts = []
    for cand in resp.get("candidates") or []:
        content = cand.get("content") or {}
        for part in content.get("parts") or []:
            if part.get("text"):
                parts.append(part["text"])
    return "".join(parts)


def gemini_collect_into_cache(client, job, specs_by_custom_id: dict, cache_dir, batch_dir) -> dict:
    """Download the result JSONL and write successes into the content-addressed cache.

    Every line is keyed back to its spec by ``key`` -- never by position. Failed lines are
    left uncached so a re-run retries exactly those, for free.
    """
    batch_dir = Path(batch_dir)
    batch_dir.mkdir(parents=True, exist_ok=True)
    dest = getattr(job, "dest", None)
    out_name = getattr(dest, "file_name", None)
    if not out_name:
        raise RuntimeError(
            f"batch {job.name} has no output file (state={_gemini_state(job)}, "
            f"error={getattr(job, 'error', None)})"
        )

    raw = with_backoff(client.files.download, file=out_name)
    text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    (batch_dir / "batch_output.jsonl").write_text(text)

    n_ok = n_err = n_unknown = 0
    errors: list[dict] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cid = row.get("key")
        spec = specs_by_custom_id.get(cid)
        if spec is None:
            n_unknown += 1
            continue
        if row.get("error") or not row.get("response"):
            n_err += 1
            errors.append({"key": cid, "error": str(row.get("error"))[:300]})
            continue
        resp = row["response"]
        body = _gemini_text_from_response(resp)
        usage = _usage_dict(
            resp.get("usageMetadata") or resp.get("usage_metadata"), GENERATE_ENDPOINT
        )
        cache.put(cache_dir, spec.cache_key(), _record(spec, body, usage, "batch", job.name))
        n_ok += 1

    summary = {
        "ok": n_ok,
        "errors": n_err,
        "unknown_keys": n_unknown,
        "batch_name": job.name,
        "state": _gemini_state(job),
        "provider": "gemini",
    }
    (batch_dir / "collect_summary.json").write_text(json.dumps(summary, indent=2))
    if errors:
        (batch_dir / "batch_errors.json").write_text(json.dumps(errors, indent=2))
    log.info("collected gemini batch %s: %d ok, %d errors", job.name, n_ok, n_err)
    return summary
