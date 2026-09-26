"""Step 1: elicit the pairwise judgments of one run's judge.

    python scripts/01_judge.py collect --run all          # offline: rebuild from the cache
    python scripts/01_judge.py smoke   --run <run>        # a few realtime calls, then STOP
    python scripts/01_judge.py sync    --run <run> --yes  # OpenAI / DeepSeek runs
    python scripts/01_judge.py batch   --run <run> --yes  # Anthropic runs
    python scripts/01_judge.py gemini  --run <run> --yes  # Gemini runs

    smoke    -- run N pairs realtime, print every raw + parsed response, measure real
                tokens, project the full-run cost, then STOP. Costs ~$0.01.
    sync     -- run all uncached pairs realtime across a thread pool. Requires --yes.
    gemini   -- same, over the Gemini Batch API (Files-API JSONL source). Requires --yes.
    batch    -- submit all uncached pairs as one Anthropic Message Batch, poll, and
                collect into the same cache. Half the realtime price. Requires --yes.
                Resumable: the batch id is kept in results/batch_meta.json, so a
                re-run re-attaches to the running batch instead of paying twice.
    collect  -- reassemble judgments.parquet / phrases.parquet from the cache alone
                (no API calls); runs automatically at the end of `sync` / `batch`.

Prompts come from the run's ``prompts/<corpus>.py`` (loaded by path): paper Prompts
``prm:elicit-ellipse`` and ``prm:elicit-asap``.

OFFLINE. ``collect`` makes no API call: every one of the 16,000 responses is in
``raw/cache/<run>/``, keyed by a hash of the exact request, so it rebuilds
``judgments.parquet`` / ``phrases.parquet`` byte-for-byte and fails loudly if the request
builders ever drift (a drifted request misses the cache and is counted as missing).

SPEND GATES. `sync` refuses to run without ``--yes`` and aborts if the projection exceeds
``judge.budget_gate_usd``. Every response is cached content-addressed, so re-runs and
resumed interruptions cost nothing.

SLOT CONVENTION: ``essay_first_id`` is prompt slot A; ``overall_winner == "A"`` becomes
``w_t = 1`` downstream. ``pair_id`` is the request ``custom_id``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

import _paths as _bootstrap  # noqa: F401  (sys.path bootstrap; see scripts/_paths.py)
from _config import load_prompts, resolve  # noqa: E402
from core.canonical import normalize_quality  # noqa: E402
from core.judging import cache  # noqa: E402
from core.judging.batch import (  # noqa: E402
    anthropic_collect_into_cache,
    anthropic_poll_until_done,
    anthropic_submit,
    gemini_collect_into_cache,
    gemini_poll_until_done,
    gemini_submit,
    run_concurrent,
    run_smoke,
)
from core.judging.pricing import actual_usd_by_source, estimate_usd  # noqa: E402
from core.openai_client import load_env  # noqa: E402
from core.judging.requests import (  # noqa: E402
    build_anthropic_scoring_spec,
    build_deepseek_scoring_spec,
    build_gemini_scoring_spec,
    build_scoring_spec,
)

ELIC = None  # the run's prompt module, loaded in main()


def _client(provider: str = "openai", key_env: str | None = None):
    """API client for the comparative judge; keys are read from ``.env`` at the repository root."""
    load_env()
    if provider == "anthropic":
        import anthropic

        return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    if provider == "gemini":
        from google import genai

        return genai.Client(api_key=os.environ[key_env or "GEMINI_API_KEY"])
    if provider == "deepseek":
        # OpenAI-compatible dialect at DeepSeek's own base_url.
        from openai import OpenAI

        return OpenAI(
            api_key=os.environ[key_env or "DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com"
        )
    if provider == "openai":
        from openai import OpenAI

        return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    raise ValueError(f"unknown judge.comparative_provider {provider!r}")


def _paths(run):
    run.results.mkdir(parents=True, exist_ok=True)
    run.cache.mkdir(parents=True, exist_ok=True)
    return run.graph, run.results, run.cache


def build_specs(cfg, graph_dir: Path) -> list:
    """One RequestSpec per pair. Deterministic: re-running recovers identical cache keys."""
    pairs = pd.read_parquet(graph_dir / "pairs.parquet")
    essays = pd.read_parquet(graph_dir / "essays.parquet")
    text = dict(zip(essays["essay_id"].astype(str), essays["full_text"].astype(str)))

    jc = cfg["judge"]
    model = jc["comparative_model"]
    effort = jc.get("judge_effort") or None
    max_out = int(jc.get("max_out_comparative", 1500))
    provider = jc.get("comparative_provider", "openai")

    specs = []
    for r in pairs.itertuples(index=False):
        a_id, b_id = str(r.essay_first_id), str(r.essay_second_id)
        system, user = ELIC.comparative_messages(text[a_id], text[b_id])
        meta = {"a_id": a_id, "b_id": b_id, "score_gap": float(r.score_gap)}
        if provider == "gemini":
            specs.append(
                build_gemini_scoring_spec(
                    custom_id=str(r.pair_id),
                    model=model,
                    system=system,
                    user=user,
                    json_schema=ELIC.COMPARATIVE_SCHEMA,
                    max_output_tokens=max_out,
                    thinking_level=jc.get("thinking_level", "MINIMAL"),
                    meta=meta,
                )
            )
        elif provider == "deepseek":
            specs.append(
                build_deepseek_scoring_spec(
                    custom_id=str(r.pair_id),
                    model=model,
                    system=system,
                    user=user,
                    json_schema=ELIC.COMPARATIVE_SCHEMA,
                    max_output_tokens=max_out,
                    thinking=jc.get("thinking", "disabled"),
                    meta=meta,
                )
            )
        elif provider == "anthropic":
            # Forced tool use carries the schema; there is no reasoning-effort knob.
            specs.append(
                build_anthropic_scoring_spec(
                    custom_id=str(r.pair_id),
                    model=model,
                    system=system,
                    user=user,
                    json_schema=ELIC.COMPARATIVE_SCHEMA,
                    max_output_tokens=max_out,
                    meta=meta,
                )
            )
        elif provider == "openai":
            specs.append(
                build_scoring_spec(
                    custom_id=str(r.pair_id),
                    model=model,
                    system=system,
                    user=user,
                    json_schema=ELIC.COMPARATIVE_SCHEMA,
                    max_output_tokens=max_out,
                    reasoning_effort=effort,
                    meta=meta,
                )
            )
        else:
            # Fail at build time rather than at call time.
            raise ValueError(f"unknown judge.comparative_provider {provider!r}")
    return specs


def _usage_mean(records) -> tuple[float, float]:
    us = [r.get("usage") or {} for r in records if not r.get("error")]
    if not us:
        return 2200.0, 260.0
    n = len(us)
    return (
        sum(u.get("input_tokens", 0) for u in us) / n,
        sum(u.get("output_tokens", 0) for u in us) / n,
    )


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------


def cmd_smoke(run, client, args) -> int:
    cfg = run.cfg
    graph, res, cache_dir = _paths(run)
    specs = build_specs(cfg, graph)
    n = int(args.n or cfg["judge"].get("smoke_pairs", 8))
    sample = specs[:n]

    print(f"smoke: {n} pairs realtime on {cfg['judge']['comparative_model']}\n")
    recs = run_smoke(client, sample, cache_dir)

    n_ok = 0
    n_qual = 0
    for spec, rec in zip(sample, recs):
        print("=" * 78)
        print(
            f"{spec.custom_id}  a={spec.meta['a_id']}  b={spec.meta['b_id']}  "
            f"score_gap={spec.meta['score_gap']}"
        )
        if rec.get("error"):
            print("  ERROR:", rec["error"])
            continue
        print("--- raw ---")
        print(rec["text"][:1400])
        parsed = ELIC.parse_comparative(rec["text"])
        print("--- parsed ---")
        if parsed is None:
            print("  PARSE FAILED")
            continue
        n_ok += 1
        n_qual += len(parsed["qualities"])
        print(f"  overall_winner: {parsed['overall_winner']}")
        for q in parsed["qualities"]:
            print(f"    - [{q['evidence']}] {q['quality']}: {q['description']}")

    in_tok, out_tok = _usage_mean(recs)
    total = len(specs)
    proj = estimate_usd(
        {cfg["judge"]["comparative_model"]: total}, in_tok=in_tok, out_tok=out_tok, batch=False
    )
    gate = float(cfg["judge"].get("budget_gate_usd", 10.0))

    print("\n" + "=" * 78)
    print("MEASURED COST")
    print("=" * 78)
    print(f"  parsed            : {n_ok}/{n}")
    print(f"  qualities/pair    : {n_qual / max(n_ok, 1):.2f}")
    print(f"  tokens/pair       : {in_tok:.0f} in / {out_tok:.0f} out")
    print(f"  full run          : {total} pairs")
    print(f"  PROJECTED (sync)  : ${proj:.2f}   [gate ${gate:.2f}]")

    report = {
        "n_smoke": n,
        "n_parsed": n_ok,
        "mean_qualities_per_pair": n_qual / max(n_ok, 1),
        "mean_input_tokens": in_tok,
        "mean_output_tokens": out_tok,
        "n_pairs_full": total,
        "projected_usd_sync": proj,
        "budget_gate_usd": gate,
        "model": cfg["judge"]["comparative_model"],
    }
    (res / "smoke_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {res / 'smoke_report.json'}")

    if n_ok < n:
        print(
            "\nSMOKE FAILED: not every pair parsed. Fix the prompt/schema before proceeding.",
            file=sys.stderr,
        )
        return 1
    print(
        f"\nSTOP for go-ahead on the full run:  01_judge.py <sync|batch|gemini> --run {run.name} --yes"
    )
    return 0


# ---------------------------------------------------------------------------
# sync
# ---------------------------------------------------------------------------


def cmd_sync(run, client, args) -> int:
    cfg = run.cfg
    graph, res, cache_dir = _paths(run)
    specs = build_specs(cfg, graph)
    cached, uncached = cache.partition_cached(cache_dir, specs)
    print(f"pairs: {len(specs)} total | {len(cached)} cached | {len(uncached)} to run")

    if uncached:
        prior = [r for r in cached.values() if not r.get("error")]
        in_tok, out_tok = _usage_mean(prior) if prior else (2200.0, 260.0)
        proj = estimate_usd(
            {cfg["judge"]["comparative_model"]: len(uncached)},
            in_tok=in_tok,
            out_tok=out_tok,
            batch=False,
        )
        gate = float(cfg["judge"].get("budget_gate_usd", 10.0))
        print(
            f"projection: ${proj:.2f} at {in_tok:.0f}/{out_tok:.0f} tokens per pair "
            f"[gate ${gate:.2f}]"
        )
        if proj > gate:
            print(f"ABORT: projection ${proj:.2f} exceeds gate ${gate:.2f}", file=sys.stderr)
            return 2
        if not args.yes:
            print("ABORT: pass --yes to spend.", file=sys.stderr)
            return 3

        workers = int(cfg["judge"].get("sync_workers", 16))
        print(f"running {len(uncached)} pairs at concurrency {workers} ...")
        _, errors = run_concurrent(
            client, uncached, cache_dir, max_workers=workers, progress_every=100
        )
        if errors:
            print(f"{len(errors)} hard errors (not cached; re-run to retry):", file=sys.stderr)
            for cid, err in errors[:10]:
                print(f"  {cid}: {err}", file=sys.stderr)

    return cmd_collect(run, client, args)


# ---------------------------------------------------------------------------
# batch  (Anthropic Message Batches -- 50% of realtime price)
# ---------------------------------------------------------------------------


def cmd_batch(run, client, args) -> int:
    cfg = run.cfg
    jc = cfg["judge"]
    if jc.get("comparative_provider") != "anthropic":
        print(
            "ABORT: `batch` is the Anthropic path; set judge.comparative_provider: anthropic",
            file=sys.stderr,
        )
        return 2

    graph, res, cache_dir = _paths(run)
    specs = build_specs(cfg, graph)
    cached, uncached = cache.partition_cached(cache_dir, specs)
    print(f"pairs: {len(specs)} total | {len(cached)} cached | {len(uncached)} to run")

    meta_path = res / "batch_meta.json"
    batch_dir = res / "batch"

    if uncached:
        prior = [r for r in cached.values() if not r.get("error")]
        in_tok, out_tok = _usage_mean(prior) if prior else (2200.0, 260.0)
        proj = estimate_usd(
            {jc["comparative_model"]: len(uncached)}, in_tok=in_tok, out_tok=out_tok, batch=True
        )
        gate = float(jc.get("budget_gate_usd", 10.0))
        print(
            f"projection: ${proj:.2f} at {in_tok:.0f}/{out_tok:.0f} tokens per pair "
            f"[gate ${gate:.2f}, batch price]"
        )
        if proj > gate:
            print(f"ABORT: projection ${proj:.2f} exceeds gate ${gate:.2f}", file=sys.stderr)
            return 2
        if not args.yes:
            print("ABORT: pass --yes to spend.", file=sys.stderr)
            return 3

    # Re-attach to an in-flight batch rather than paying for a second one.
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    batch_id = (meta or {}).get("batch_id")
    if batch_id and uncached:
        print(f"re-attaching to existing batch {batch_id} (delete {meta_path.name} to resubmit)")
    elif uncached:
        meta = anthropic_submit(client, uncached, meta_path)
        batch_id = meta["batch_id"]
        print(f"submitted batch {batch_id} ({len(uncached)} requests)")

    if batch_id:
        print("polling (30s interval; ctrl-C is safe -- re-run to resume) ...")
        batch = anthropic_poll_until_done(
            client,
            batch_id,
            interval_s=30,
            max_wait_s=int(jc.get("batch_timeout_s", 86400)),
            meta_path=meta_path,
        )
        print(f"batch {batch_id} finished: {batch.processing_status}")
        summary = anthropic_collect_into_cache(
            client, batch_id, {s.custom_id: s for s in specs}, cache_dir, batch_dir
        )
        print(f"collected: {summary['ok']} ok, {summary['errors']} errors")
        if summary["errors"]:
            print(
                f"  failed requests are NOT cached; re-run `batch` to retry only those "
                f"(see {batch_dir / 'batch_errors.json'})",
                file=sys.stderr,
            )

    return cmd_collect(run, client, args)


# ---------------------------------------------------------------------------
# gemini  (Gemini Batch API -- 50% of realtime price)
# ---------------------------------------------------------------------------


def cmd_gemini(run, client, args) -> int:
    cfg = run.cfg
    jc = cfg["judge"]
    if jc.get("comparative_provider") != "gemini":
        print(
            "ABORT: `gemini` is the Google path; set judge.comparative_provider: gemini",
            file=sys.stderr,
        )
        return 2

    graph, res, cache_dir = _paths(run)
    specs = build_specs(cfg, graph)
    cached, uncached = cache.partition_cached(cache_dir, specs)
    print(f"pairs: {len(specs)} total | {len(cached)} cached | {len(uncached)} to run")

    batch_dir = res / "batch"
    batch_dir.mkdir(parents=True, exist_ok=True)

    if uncached:
        prior = [r for r in cached.values() if not r.get("error")]
        in_tok, out_tok = _usage_mean(prior) if prior else (2300.0, 400.0)
        proj = estimate_usd(
            {jc["comparative_model"]: len(uncached)}, in_tok=in_tok, out_tok=out_tok, batch=True
        )
        gate = float(jc.get("budget_gate_usd", 10.0))
        print(
            f"projection: ${proj:.2f} at {in_tok:.0f}/{out_tok:.0f} tokens per pair "
            f"[gate ${gate:.2f}, batch price]"
        )
        if proj > gate:
            print(f"ABORT: projection ${proj:.2f} exceeds gate ${gate:.2f}", file=sys.stderr)
            return 2
        if not args.yes:
            print("ABORT: pass --yes to spend.", file=sys.stderr)
            return 3

    # Chunked and SEQUENTIAL. One 1,992-request job is refused with 429 RESOURCE_EXHAUSTED
    # (a 1-request job on the same key is accepted, so this is an enqueued-work quota, not
    # a batch entitlement). Chunks are submitted, polled and collected one at a time; the
    # content-addressed cache makes the whole loop resumable, since a re-run re-partitions
    # and only the still-uncached pairs are re-chunked.
    by_id = {sp.custom_id: sp for sp in specs}
    size = int(jc.get("batch_chunk_size", 250))
    chunks = [uncached[i : i + size] for i in range(0, len(uncached), size)]
    print(f"submitting {len(chunks)} chunk(s) of up to {size} requests, sequentially")

    total_ok = total_err = 0
    for idx, chunk in enumerate(chunks):
        # Key the meta file on the chunk's CONTENT, not its position. A resumed run
        # re-partitions `uncached`, so chunk i is generally NOT the same set of pairs it
        # was last time -- and a positional key would re-attach to a stale batch, re-collect
        # already-cached pairs and silently never submit the outstanding ones. Hashing the
        # custom_ids makes re-attachment exact: same pairs -> same job, different pairs ->
        # fresh submission.
        digest = hashlib.sha256("\n".join(sp.custom_id for sp in chunk).encode()).hexdigest()[:12]
        cmeta = batch_dir / f"meta_{digest}.json"
        legacy = batch_dir / f"meta_{idx:02d}.json"
        if not cmeta.exists() and legacy.exists():
            # Position-keyed meta file: only valid if that job covered exactly this chunk.
            prior_ids = json.loads(legacy.read_text()).get("custom_ids")
            if prior_ids == [sp.custom_id for sp in chunk]:
                cmeta = legacy
        prior = json.loads(cmeta.read_text()) if cmeta.exists() else {}
        name = prior.get("batch_name")
        if name:
            print(f"[chunk {idx + 1}/{len(chunks)}] re-attaching to {name}")
        else:
            m = gemini_submit(
                client,
                chunk,
                batch_dir / f"requests_{idx:02d}.jsonl",
                cmeta,
                display_name=f"interp-scoring-gemini-{digest}",
            )
            name = m["batch_name"]
            print(f"[chunk {idx + 1}/{len(chunks)}] submitted {name} ({len(chunk)} requests)")

        job = gemini_poll_until_done(
            client,
            name,
            interval_s=int(jc.get("batch_poll_s", 20)),
            max_wait_s=int(jc.get("batch_timeout_s", 86400)),
            meta_path=cmeta,
        )
        summary = gemini_collect_into_cache(
            client, job, by_id, cache_dir, batch_dir / f"chunk_{digest}"
        )
        total_ok += summary["ok"]
        total_err += summary["errors"]
        print(
            f"[chunk {idx + 1}/{len(chunks)}] {summary['ok']} ok, {summary['errors']} errors "
            f"(running total {total_ok} ok)"
        )

    if chunks:
        print(f"\nall chunks done: {total_ok} ok, {total_err} errors")
        if total_err:
            print(
                "  failed requests are NOT cached; re-run `gemini` to retry only those",
                file=sys.stderr,
            )

    return cmd_collect(run, client, args)


# ---------------------------------------------------------------------------
# collect  (no API calls -- cache only)
# ---------------------------------------------------------------------------


def cmd_collect(run, client, args) -> int:
    cfg = run.cfg
    graph, res, cache_dir = _paths(run)
    specs = build_specs(cfg, graph)
    pairs = pd.read_parquet(graph / "pairs.parquet").set_index("pair_id")

    jrows, prows, missing, records = [], [], 0, []
    for spec in specs:
        rec = cache.get(cache_dir, spec.cache_key())
        if rec is None or rec.get("error"):
            missing += 1
            continue
        records.append(rec)
        parsed = ELIC.parse_comparative(rec.get("text", ""))
        pid = spec.custom_id
        a_id, b_id = spec.meta["a_id"], spec.meta["b_id"]
        u = rec.get("usage") or {}

        if parsed is None:
            jrows.append(
                {
                    "pair_id": pid,
                    "essay_first_id": a_id,
                    "essay_second_id": b_id,
                    "winner": None,
                    "qualities": None,
                    "parse_failed": True,
                    "n_qualities": 0,
                    "score_gap": spec.meta["score_gap"],
                    "input_tokens": u.get("input_tokens", 0),
                    "output_tokens": u.get("output_tokens", 0),
                }
            )
            continue

        quals = [q["quality"] for q in parsed["qualities"]]
        jrows.append(
            {
                "pair_id": pid,
                "essay_first_id": a_id,
                "essay_second_id": b_id,
                # `winner` in the canonical schema build_tensors consumes: "A" -> w_t = 1.
                "winner": parsed["overall_winner"],
                "qualities": quals,
                "parse_failed": False,
                "n_qualities": len(quals),
                "score_gap": spec.meta["score_gap"],
                "input_tokens": u.get("input_tokens", 0),
                "output_tokens": u.get("output_tokens", 0),
            }
        )
        for q in parsed["qualities"]:
            prows.append(
                {
                    "pair_id": pid,
                    "phrase": q["quality"],
                    "phrase_norm": normalize_quality(q["quality"]),
                    "description": q["description"],
                    "evidence": q["evidence"],
                    "winner_slot": parsed["overall_winner"],
                }
            )

    judgments = pd.DataFrame(jrows)
    phrases = pd.DataFrame(prows)
    judgments.to_parquet(res / "judgments.parquet", index=False)
    phrases.to_parquet(res / "phrases.parquet", index=False)

    spend = actual_usd_by_source(records)
    n_batched = sum(1 for r in records if r.get("source") == "batch")
    ledger = {
        "model": cfg["judge"]["comparative_model"],
        "provider": cfg["judge"].get("comparative_provider", "openai"),
        "mode": "batch" if n_batched > len(records) / 2 else "sync",
        "n_batched_records": n_batched,
        "n_pairs": len(judgments),
        "n_cached_records": len(records),
        "input_tokens": int(judgments["input_tokens"].sum()),
        "output_tokens": int(judgments["output_tokens"].sum()),
        "realized_usd": spend,
    }
    (res / "cost_ledger.json").write_text(json.dumps(ledger, indent=2))

    n_fail = int(judgments["parse_failed"].sum()) if len(judgments) else 0
    ok = judgments[~judgments["parse_failed"]] if len(judgments) else judgments
    print(f"\ncollected {len(judgments)} pairs ({missing} missing from cache)")
    print(f"  parse failures : {n_fail} ({100 * n_fail / max(len(judgments), 1):.2f}%)")
    if len(ok):
        print(
            f"  qualities/pair : {ok['n_qualities'].mean():.2f} "
            f"(min {ok['n_qualities'].min()}, max {ok['n_qualities'].max()})"
        )
        print(f"  slot-A wins    : {(ok['winner'] == 'A').mean():.3f}")
        print(f"  unique phrases : {phrases['phrase_norm'].nunique()}")
    print(f"  realized cost  : ${spend:.2f}")
    print(f"\nwrote → {res}")
    if missing:
        print(f"INCOMPLETE: {missing} pairs have no usable cached response", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    global ELIC
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["smoke", "sync", "batch", "gemini", "collect"])
    ap.add_argument(
        "--run", required=True, help="a run name; `collect` also takes a comma list or 'all'"
    )
    ap.add_argument("--n", type=int, default=None, help="smoke: number of pairs")
    ap.add_argument("--yes", action="store_true", help="confirm spending (sync/batch/gemini)")
    args = ap.parse_args()

    runs = resolve(args.run)
    if args.command != "collect" and len(runs) != 1:
        ap.error(f"`{args.command}` spends money: name exactly one run")
    status = 0
    for run in runs:
        print(f"\n=== {run.name}")
        ELIC = load_prompts(run)
        provider = run.cfg["judge"].get("comparative_provider", "openai")
        client = (
            None
            if args.command == "collect"
            else _client(provider, run.cfg["judge"].get("api_key_env"))
        )
        status |= {
            "smoke": cmd_smoke,
            "sync": cmd_sync,
            "batch": cmd_batch,
            "gemini": cmd_gemini,
            "collect": cmd_collect,
        }[args.command](run, client, args)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
