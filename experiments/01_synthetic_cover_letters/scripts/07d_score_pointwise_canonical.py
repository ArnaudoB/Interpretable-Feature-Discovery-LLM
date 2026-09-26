"""Step 7d: score all 200 letters pointwise against OUR canonicalized criteria (Batch API).

    python scripts/07d_score_pointwise_canonical.py smoke   --config config.yaml
    python scripts/07d_score_pointwise_canonical.py submit  --config config.yaml
    python scripts/07d_score_pointwise_canonical.py collect --config config.yaml

The ablation of the statistical model, and the source of tab:main's four "Ablations" rows.
The criteria are held fixed at what canonicalizing the pairwise rationales produced (step
03); only the SCORING mechanism changes -- a 0-10 absolute rating per criterion instead of
the gated Bradley-Terry fit of step 04. Two arms:

  k32  every canonical criterion but the ``Other`` catch-all -- the panel you have if you
       never fit the model at all, since the kappa/rho gate IS part of the model
  k21  the gate-kept set -- isolates the score estimates from the gate

so k32-vs-k21 separates the gate's contribution from the score estimates'. Mapping onto the
paper, where the arms are named for what they ablate rather than for K:

  Pairwise comparisons + LLM scoring              k32, rep 0
  Pairwise comparisons + LLM scoring--Multi-score k32, mean over n* reps
  Reliable features + LLM scoring                 k21, rep 0
  Reliable features + LLM scoring--Multi-score    k21, mean over n* reps

The rep count n* = floor(comparative_budget / per_rep) hands each pointwise ensemble what the
comparative pass spent on LLM calls; it is fixed in config (``pointwise_canonical.n_star``).
``smoke`` measures per-rep token cost and re-solves n* for comparison (the budget is read from
the response cache, so it tracks the price table). ``submit`` builds 200 x n* specs (variants cycled, criterion order permuted, seeded) and
submits one 24h batch per arm. ``collect`` assembles scores.parquet (letter x criterion, mean
over reps), scores_long.parquet and cost_ledger.json.

The 1-rep arms are rep 0 of the same long frame, so they cost nothing extra -- exactly as M1
is read off M2's frame in 07_score_pointwise.py. Cached under ``cache/responses`` (shipped),
so all three commands replay offline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import _paths  # noqa: F401  (sys.path bootstrap; see scripts/_paths.py)
from core.judging import cache
from core.judging.batch import (
    build_jsonl,
    submit_batch,
    poll_until_done,
    collect_into_cache,
    run_smoke,
)
from core.judging.cache import partition_cached
from core.judging.requests import build_scoring_spec, RESPONSES_ENDPOINT
from core.judging.pricing import price_for, actual_usd, BATCH_DISCOUNT
from core.openai_client import get_client
import prompts_pointwise_canonical as PWC

ARMS = ("k32", "k21")


def _root(args):
    return Path(args.config).parent


def _sub(arm):
    return f"pointwise_canonical_{arm}"


def _out_dir(root, arm, override=None):
    """Where the arm writes. ``override`` (--out-dir) lets a replay land in a scratch tree so
    the shipped artifacts can be regenerated and diffed rather than overwritten."""
    d = (Path(override) / _sub(arm)) if override else (root / "results" / _sub(arm))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _text(cfg, root):
    essays = pd.read_parquet(root / cfg["graph"]["out_dir"] / "essays.parquet")
    return dict(zip(essays["letter_id"], essays["text"]))


def _criteria(cfg, root, arm):
    """The arm's criterion panel, in a fixed (name-sorted) canonical order.

    k32 drops the ``Other`` catch-all, which has no single construct to rate. k21 reads the
    kappa/rho-kept set off the fit summary, so the panel is exactly the one tab:main reports.
    """
    pc = cfg["pointwise_canonical"]
    res = root / "results"
    crit = json.loads((res / pc["criteria_source"]).read_text())["criteria"]
    if arm == "k21":
        kept = set(json.loads((res / pc["gated_fit_summary"]).read_text())["kept"])
        crit = [c for c in crit if c["name"] in kept]
    else:
        drop = set(pc.get("drop_criteria", [PWC._p.OTHER_CRITERION]))
        crit = [c for c in crit if c["name"] not in drop]
    return sorted(crit, key=lambda c: c["name"])


def _score_specs(cfg, text, letter_ids, criteria, reps, arm):
    """One spec per (letter, rep); variant cycled, criterion order permuted, deterministic."""
    j = cfg["judge"]
    variants = PWC.POINTWISE_INSTRUCTION_VARIANTS
    base = int(j.get("seed", 42))
    max_out = int(cfg["pointwise_canonical"].get("max_out", j["max_out_pointwise"]))
    specs = []
    for lid in letter_ids:
        for rep in range(reps):
            v = variants[rep % len(variants)]
            seed = int(
                hashlib.md5(f"{base}:pwcanon:{arm}:{lid}:{rep}".encode()).hexdigest()[:8], 16
            )
            rng = np.random.default_rng(seed)
            perm = [criteria[i] for i in rng.permutation(len(criteria))]
            system, user = PWC.pointwise_canonical_messages(text[lid], perm, v)
            specs.append(
                build_scoring_spec(
                    custom_id=f"pwcanon_{arm}:{lid}:r{rep:02d}",
                    model=j["pointwise_model"],
                    system=system,
                    user=user,
                    json_schema=PWC.POINTWISE_CANONICAL_SCHEMA,
                    max_output_tokens=max_out,
                    reasoning_effort=j["judge_effort"],
                    meta={"letter_id": lid, "rep": rep, "variant": v["key"], "arm": arm},
                )
            )
    return specs


def _comparative_budget(cfg, root):
    """The comparative pass's realized LLM spend -- the budget these arms are handed.

    Read from the cache rather than a stored constant, so it tracks
    ``core/judging/pricing.py`` instead of going stale against it.
    """
    cache_dir = root / cfg["judge"]["cache_dir"]
    recs = []
    for p in (cache_dir / "responses").glob("*/*.json"):
        try:
            r = json.loads(p.read_text())
        except (ValueError, OSError):
            continue
        if (r.get("custom_id") or "").startswith("cmp:"):
            recs.append(r)
    return actual_usd(recs, cfg.get("pricing"), batch=True), len(recs)


def _n_star(cfg, arm):
    """The arm's rep count, fixed in config (``pointwise_canonical.n_star``), never re-solved.

    ``smoke`` prints the value re-solved under the current price table for comparison only.
    """
    n = cfg["pointwise_canonical"]["n_star"][arm]
    return int(n)


def cmd_smoke(cfg, root, client, arm, out_dir=None):
    j = cfg["judge"]
    text = _text(cfg, root)
    crits = _criteria(cfg, root, arm)
    names = [c["name"] for c in crits]
    ids = sorted(text)[: int(j["smoke_letters"])]
    specs = _score_specs(cfg, text, ids, crits, len(PWC.POINTWISE_INSTRUCTION_VARIANTS), arm)
    recs = run_smoke(client, specs, root / j["cache_dir"])
    pin = float(np.mean([r["usage"].get("input_tokens", 0) for r in recs]))
    pout = float(np.mean([r["usage"].get("output_tokens", 0) for r in recs]))
    parsed = [PWC.parse_pointwise(r["text"], names) for r in recs]
    complete = float(np.mean([len(p) == len(names) for p in parsed]))
    allsc = [v for p in parsed for v in p.values()]
    pin_p, pout_p = price_for(j["pointwise_model"], cfg.get("pricing"))
    per_spec = (pin * pin_p + pout * pout_p) / 1e6 * BATCH_DISCOUNT
    per_rep = len(text) * per_spec
    budget, n_cmp = _comparative_budget(cfg, root)
    solved = max(1, int(budget // per_rep))
    pinned = _n_star(cfg, arm)
    print(
        f"[smoke {arm}] K={len(names)} n={len(recs)} in~{pin:.0f} out~{pout:.0f} | "
        f"complete panels {complete:.0%}"
    )
    print(
        f"[smoke {arm}] scores: mean {np.mean(allsc):.2f} sd {np.std(allsc):.2f} "
        f"min {min(allsc)} max {max(allsc)} | share in 5-7: "
        f"{np.mean([5 <= v <= 7 for v in allsc]):.0%}"
    )
    print(f"[smoke {arm}] ${per_spec:.6f}/spec  ${per_rep:.3f}/rep (200 letters, batch)")
    print(
        f"[smoke {arm}] comparative budget ${budget:.3f} ({n_cmp} calls) -> re-solved n* = "
        f"{solved}; configured n* = {pinned} (scoring ${pinned * per_rep:.2f})"
    )
    if solved != pinned:
        print(
            f"[smoke {arm}] NOTE re-solving under this price table gives n* = {solved}; "
            f"`submit` uses the configured n* = {pinned}."
        )
    (_out_dir(root, arm, out_dir) / "smoke_report_replay.json").write_text(
        json.dumps(
            {
                "arm": arm,
                "K": len(names),
                "in": pin,
                "out": pout,
                "complete_panel_rate": complete,
                "score_mean": float(np.mean(allsc)),
                "score_sd": float(np.std(allsc)),
                "share_5_to_7": float(np.mean([5 <= v <= 7 for v in allsc])),
                "usd_per_spec_batch": per_spec,
                "usd_per_rep_batch": per_rep,
                "comparative_budget_usd": budget,
                "n_comparative_calls": n_cmp,
                "n_star_pinned": pinned,
                "n_star_resolved_here": solved,
                "projected_scoring_usd": pinned * per_rep,
            },
            indent=2,
        )
    )
    print(
        f"[smoke {arm}] wrote results/{_sub(arm)}/smoke_report_replay.json "
        f"(the shipped smoke_report.json is the run of record and is not overwritten)"
    )


def cmd_submit(cfg, root, client, arm, out_dir=None):
    text = _text(cfg, root)
    crits = _criteria(cfg, root, arm)
    R = _n_star(cfg, arm)
    specs = _score_specs(cfg, text, sorted(text), crits, R, arm)
    cached, todo = partition_cached(root / cfg["judge"]["cache_dir"], specs)
    bd = root / "batch" / _sub(arm)
    bd.mkdir(parents=True, exist_ok=True)
    print(
        f"[submit {arm}] {len(specs)} specs (200 x n*={R}, K={len(crits)}) | "
        f"cached {len(cached)} | to submit {len(todo)}"
    )
    (bd / "run_params.json").write_text(
        json.dumps({"n_star": R, "K": len(crits), "arm": arm}, indent=2)
    )
    if not todo:
        (bd / "meta.json").write_text(json.dumps({"batch_id": None, "status": "all_cached"}))
        return
    build_jsonl(todo, bd / "requests.jsonl")
    submit_batch(client, bd / "requests.jsonl", RESPONSES_ENDPOINT, bd / "meta.json")
    print(f"[submit {arm}] submitted a 24h batch — re-run `collect` when it resolves.")


def cmd_collect(cfg, root, client, arm, out_dir=None):
    text = _text(cfg, root)
    crits = _criteria(cfg, root, arm)
    names = [c["name"] for c in crits]
    bd = root / "batch" / _sub(arm)
    rp = bd / "run_params.json"
    R = int(json.loads(rp.read_text())["n_star"]) if rp.exists() else _n_star(cfg, arm)
    specs = _score_specs(cfg, text, sorted(text), crits, R, arm)
    cache_dir = root / cfg["judge"]["cache_dir"]
    meta = json.loads((bd / "meta.json").read_text()) if (bd / "meta.json").exists() else {}
    if meta.get("batch_id"):
        client = client or get_client()
        batch = poll_until_done(client, meta["batch_id"], interval_s=60, meta_path=bd / "meta.json")
        collect_into_cache(client, batch, {s.custom_id: s for s in specs}, cache_dir, bd)

    recs = {s.custom_id: cache.get(cache_dir, s.cache_key()) for s in specs}
    present = [
        (s, recs[s.custom_id]) for s in specs if recs[s.custom_id] and recs[s.custom_id].get("text")
    ]
    rows, partial = [], 0
    for s, r in present:
        m = s.meta
        scored = PWC.parse_pointwise(r["text"], names)
        if len(scored) != len(names):
            partial += 1
        for crit, sc in scored.items():
            rows.append(
                {
                    "letter_id": m["letter_id"],
                    "rep": m["rep"],
                    "variant": m["variant"],
                    "dimension": crit,
                    "score": sc,
                }
            )
    out = _out_dir(root, arm, out_dir)
    long = pd.DataFrame(rows)
    long.to_parquet(out / "scores_long.parquet", index=False)
    long.groupby(["letter_id", "dimension"])["score"].mean().unstack().to_parquet(
        out / "scores.parquet"
    )
    usd = actual_usd([r for _, r in present], cfg.get("pricing"), batch=True)
    (out / "cost_ledger.json").write_text(
        json.dumps(
            {
                "arm": arm,
                "n_specs": len(specs),
                "n_present": len(present),
                "n_star": R,
                "n_partial_panels": partial,
                "n_letters": int(long["letter_id"].nunique()),
                "n_dims": int(long["dimension"].nunique()),
                "reported_basis": "batch",
                "realized_usd_batch": usd,
            },
            indent=2,
        )
    )
    print(
        f"[collect {arm}] {len(present)}/{len(specs)} present, {partial} partial panels -> "
        f"{long['letter_id'].nunique()} letters x {long['dimension'].nunique()} criteria | "
        f"${usd:.2f} batch"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=["smoke", "submit", "collect"])
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--arm", default="all", choices=("all",) + ARMS)
    p.add_argument(
        "--out-dir",
        default=None,
        help="write this run's artifacts under DIR/<arm>/ instead of results/<arm>/, "
        "so a replay can be diffed against the shipped run of record",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="required by `smoke` and `submit`, which can spend money; `collect` "
        "is a pure cache read when every spec is present (the shipped case)",
    )
    args = p.parse_args()
    if args.command in ("smoke", "submit") and not args.yes:
        raise SystemExit(
            f"`{args.command}` can issue paid API calls — pass --yes to confirm. "
            f"Everything the paper reports replays from the shipped cache with "
            f"`collect`, which needs no key and spends nothing."
        )
    cfg = yaml.safe_load(Path(args.config).read_text())
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # `collect` only needs a client to fetch a pending batch; replaying the cache needs no key.
    client = None if args.command == "collect" else get_client()
    fn = {"smoke": cmd_smoke, "submit": cmd_submit, "collect": cmd_collect}[args.command]
    for arm in ARMS if args.arm == "all" else (args.arm,):
        fn(cfg, _root(args), client, arm, out_dir=args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
