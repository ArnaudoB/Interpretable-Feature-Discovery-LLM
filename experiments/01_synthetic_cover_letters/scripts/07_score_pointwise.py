"""Step 7: score all 200 letters against the pointwise rubrics (Batch API).

    python scripts/07_score_pointwise.py submit  --config config.yaml   # launch the scoring batches
    python scripts/07_score_pointwise.py collect --config config.yaml   # poll + assemble scores

Scores every letter against each rubric on each dimension's own 1..N scale (the four
reasoning-path variants cycled across reps, rubric order permuted, seeded). Produces the
pointwise arms of tab:main (1a/1b; 2a/2b from the taxonomy fusion) and the unreported M3a/b/c:

  * results/pointwise/       <- full_dataset_rubric.json,  R reps  (M1 = one rep, M2 = mean over reps)
  * results/pointwise_3a/    <- pooled_rubric_3a.json,     R3 reps (M3a)
  * results/pointwise_3b/    <- pooled_rubric_3b.json,     R3 reps (M3b)
  * results/pointwise_3c/    <- pooled_rubric_3c.json,     R3 reps (M3c)

Each writes scores.parquet (letter x dimension, mean over reps), scores_long.parquet, and
cost_ledger.json. Cached under ``cache/responses`` (shipped).
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
    run_concurrent,
)
from core.judging.cache import partition_cached
from core.judging.requests import build_scoring_spec, RESPONSES_ENDPOINT
from core.judging.pricing import actual_usd
import prompts as ELIC
from core.openai_client import get_client

_SCORE_SYS = "You score a cover letter against a fixed rubric and output only JSON."

# (rubric file relative to results/, output namespace, reps). Full-dataset rubric = M1/M2;
# pooled rubrics = M3a/b/c (fewer reps: their rubric elicitation already dominates cost).
# The taxonomy-fused rubric (06b, canonical run 000) is scored at n* reps -- the count that
# cost-matches "Ours" on a batch basis, solved by 06d. Its single-rep (M1-comparable) variant is
# read off rep 0 of the same long frame, exactly as M1 is read off M2's, so it costs nothing
# extra. ``"n*"`` means "read n_star from taxo_pool/cost_match.json at run time".
ARMS = [
    ("rubric_stability/full_dataset_rubric.json", "pointwise", None),  # reps from config.judge.reps
    ("rubric_stability/pooled_rubric_3a.json", "pointwise_3a", 8),
    ("rubric_stability/pooled_rubric_3b.json", "pointwise_3b", 8),
    ("rubric_stability/pooled_rubric_3c.json", "pointwise_3c", 8),
    ("rubric_stability/taxo_pool/run_000.json", "pointwise_taxo", "n*"),
]


def _root(args):
    return Path(args.config).parent


def _text(cfg, root):
    essays = pd.read_parquet(root / cfg["graph"]["out_dir"] / "essays.parquet")
    return dict(zip(essays["letter_id"], essays["text"]))


def _score_specs(cfg, text, features, reps):
    j = cfg["judge"]
    base = int(j.get("seed", 42))
    variants = ELIC.POINTWISE_INSTRUCTION_VARIANTS
    specs = []
    for lid in sorted(text):
        for rep in range(reps):
            v = variants[rep % len(variants)]
            seed = int(hashlib.md5(f"{base}:pointwise:{lid}:{rep}".encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            perm = [features[i] for i in rng.permutation(len(features))]
            specs.append(
                build_scoring_spec(
                    custom_id=f"pw:{lid}:r{rep:02d}",
                    model=j["pointwise_model"],
                    system=_SCORE_SYS,
                    user=ELIC.pointwise_scoring_prompt(text[lid], perm, v),
                    json_schema=ELIC.scoring_schema_for(v),
                    max_output_tokens=int(j["max_out_pointwise"]),
                    reasoning_effort=j["judge_effort"],
                    meta={"letter_id": lid, "rep": rep, "variant": v["key"]},
                )
            )
    return specs


def _run_arm(cfg, root, client, rubric_rel, out_sub, reps, collect, sync=False):
    j = cfg["judge"]
    text = _text(cfg, root)
    features = json.loads((root / "results" / rubric_rel).read_text())["features"]
    if reps == "n*":
        cm = root / "results" / "rubric_stability" / "taxo_pool" / "cost_match.json"
        if not cm.exists():
            raise SystemExit("run 06d_cost_match.py first — it solves n* for the taxonomy arm")
        reps = json.loads(cm.read_text())["n_star"]
    reps = int(reps or j["reps"])
    specs = _score_specs(cfg, text, features, reps)
    cache_dir = root / j["cache_dir"]
    bd = root / "batch" / out_sub
    if collect:
        meta = (
            json.loads((bd / "meta.json").read_text())
            if (bd / "meta.json").exists()
            else {"batch_id": None}
        )
        if meta.get("batch_id"):
            batch = poll_until_done(
                client, meta["batch_id"], interval_s=60, meta_path=bd / "meta.json"
            )
            collect_into_cache(client, batch, {s.custom_id: s for s in specs}, cache_dir, bd)
    elif sync:
        # Realtime path for small arms. Cost is still *reported* at batch rates (below) so the
        # table stays on one basis; `mode` in the ledger records what was actually submitted.
        cached, todo = partition_cached(cache_dir, specs)
        print(
            f"[{out_sub}] {len(specs)} specs (200 x R={reps}) | cached {len(cached)} | "
            f"to run sync {len(todo)}"
        )
        if todo:
            _, errors = run_concurrent(client, todo, cache_dir, max_workers=8)
            if errors:
                print(
                    f"[{out_sub}] {len(errors)} hard failures (not cached — re-run to retry); "
                    f"first: {errors[0][1][:120]}"
                )
    else:
        cached, todo = partition_cached(cache_dir, specs)
        print(
            f"[{out_sub}] {len(specs)} specs (200 x R={reps}) | cached {len(cached)} | to submit {len(todo)}"
        )
        if todo:
            bd.mkdir(parents=True, exist_ok=True)
            build_jsonl(todo, bd / "requests.jsonl")
            submit_batch(client, bd / "requests.jsonl", RESPONSES_ENDPOINT, bd / "meta.json")
            print(f"[{out_sub}] submitted a 24h batch — re-run `collect` when it resolves.")
            return

    recs = {s.custom_id: cache.get(cache_dir, s.cache_key()) for s in specs}
    rows, present = [], []
    for s in specs:
        r = recs[s.custom_id]
        if not (r and r.get("text")):
            continue
        present.append(r)
        m = s.meta
        for sc in json.loads(r["text"])["scores"]:
            rows.append(
                {
                    "letter_id": m["letter_id"],
                    "rep": m["rep"],
                    "variant": m["variant"],
                    "dimension": sc["dimension"],
                    "score": sc["score"],
                }
            )
    out = root / "results" / out_sub
    out.mkdir(parents=True, exist_ok=True)
    long = pd.DataFrame(rows)
    long.to_parquet(out / "scores_long.parquet", index=False)
    long.groupby(["letter_id", "dimension"])["score"].mean().unstack().to_parquet(
        out / "scores.parquet"
    )
    usd = actual_usd(present, cfg.get("pricing"), batch=True)
    (out / "cost_ledger.json").write_text(
        json.dumps(
            {
                "n_specs": len(specs),
                "n_present": len(present),
                "R": reps,
                "n_letters": int(long["letter_id"].nunique()) if len(long) else 0,
                "n_dims": int(long["dimension"].nunique()) if len(long) else 0,
                "mode": ("sync" if sync else "batch"),
                "reported_basis": "batch",
                "realized_usd_batch": usd,
            },
            indent=2,
        )
    )
    print(
        f"[{out_sub}] {len(present)}/{len(specs)} present -> scores.parquet "
        f"({long['dimension'].nunique() if len(long) else 0} dims) | ${usd:.2f} batch"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=["submit", "collect"])
    p.add_argument("--config", default="config.yaml")
    p.add_argument(
        "--arm", default="all", help="one of pointwise/pointwise_3a/3b/3c/pointwise_taxo, or 'all'"
    )
    p.add_argument(
        "--sync",
        action="store_true",
        help="run realtime instead of Batch (small arms); cost is still reported at batch rates",
    )
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    client = get_client()
    arms = ARMS if args.arm == "all" else [a for a in ARMS if a[1] == args.arm]
    for rubric_rel, out_sub, reps in arms:
        _run_arm(
            cfg,
            _root(args),
            client,
            rubric_rel,
            out_sub,
            reps,
            collect=(args.command == "collect"),
            sync=args.sync,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
