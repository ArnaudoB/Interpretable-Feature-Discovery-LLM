"""Step 7b: score every taxonomy-fusion replicate once, for method 2a's run-variance bootstrap.

    python scripts/07b_score_taxo_boot.py submit  --config config.yaml
    python scripts/07b_score_taxo_boot.py collect --config config.yaml

``06b`` produces R independent taxonomy fusions, each over a random E-subset of the elicitations.
``07_score_pointwise.py`` scores only the canonical one (run 000) -- that gives the 2a/2b point
estimates plus scoring noise for a *fixed* fusion, which is not the quantity we want a CI on.
The variance that matters is across fusions, so this scores EVERY replicate once (1 pass, the
M1-comparable depth) and hands the per-replicate score matrices to ``08c``.

Run 000 rep 0 is already cached from ``07``: the prompt is built identically here (same rubric
order permutation, seeded on letter+rep only, not on the rubric), so it is a cache hit and only
the remaining R-1 replicates are billed.

Writes results/pointwise_taxo_boot/scores_b{NNN}.parquet + cost_ledger.json.
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
from core.judging.batch import build_jsonl, collect_into_cache, poll_until_done, submit_batch
from core.judging.cache import partition_cached
from core.judging.pricing import actual_usd
from core.judging.requests import RESPONSES_ENDPOINT, build_scoring_spec
from core.openai_client import get_client
import prompts as ELIC

_SCORE_SYS = "You score a cover letter against a fixed rubric and output only JSON."
_OUT_SUB = "pointwise_taxo_boot"
_REP = 0  # single pass per replicate; variant 0, matching M1's depth


def _text(cfg, root):
    essays = pd.read_parquet(root / cfg["graph"]["out_dir"] / "essays.parquet")
    return dict(zip(essays["letter_id"], essays["text"]))


def _specs(cfg, text, runs):
    """One scoring call per (replicate, letter). Prompt construction matches 07 exactly."""
    j = cfg["judge"]
    base = int(j.get("seed", 42))
    variant = ELIC.POINTWISE_INSTRUCTION_VARIANTS[_REP % len(ELIC.POINTWISE_INSTRUCTION_VARIANTS)]
    specs = []
    for b, payload in runs:
        features = payload["features"]
        for lid in sorted(text):
            seed = int(hashlib.md5(f"{base}:pointwise:{lid}:{_REP}".encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            perm = [features[i] for i in rng.permutation(len(features))]
            specs.append(
                build_scoring_spec(
                    custom_id=f"pwtx:b{b:03d}:{lid}",
                    model=j["pointwise_model"],
                    system=_SCORE_SYS,
                    user=ELIC.pointwise_scoring_prompt(text[lid], perm, variant),
                    json_schema=ELIC.scoring_schema_for(variant),
                    max_output_tokens=int(j["max_out_pointwise"]),
                    reasoning_effort=j["judge_effort"],
                    meta={"b": b, "letter_id": lid},
                )
            )
    return specs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=["submit", "collect"])
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    j = cfg["judge"]
    root = Path(args.config).parent
    pool = root / "results" / "rubric_stability" / "taxo_pool"
    cache_dir = root / j["cache_dir"]
    bd = root / "batch" / _OUT_SUB
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    runs = []
    for rf in sorted(pool.glob("run_*.json")):
        runs.append((int(rf.stem.split("_")[1]), json.loads(rf.read_text())))
    if not runs:
        raise SystemExit("no taxonomy runs — run 06b_pool_taxonomy.py collect first")

    text = _text(cfg, root)
    specs = _specs(cfg, text, runs)
    client = get_client()

    if args.command == "collect":
        meta = json.loads((bd / "meta.json").read_text()) if (bd / "meta.json").exists() else {}
        if meta.get("batch_id"):
            batch = poll_until_done(
                client, meta["batch_id"], interval_s=60, meta_path=bd / "meta.json"
            )
            collect_into_cache(client, batch, {s.custom_id: s for s in specs}, cache_dir, bd)
    else:
        cached, todo = partition_cached(cache_dir, specs)
        print(
            f"[{_OUT_SUB}] {len(specs)} specs ({len(runs)} replicates x {len(text)} letters) | "
            f"cached {len(cached)} | to submit {len(todo)}"
        )
        if todo:
            bd.mkdir(parents=True, exist_ok=True)
            n = build_jsonl(todo, bd / "requests.jsonl")
            mb = (bd / "requests.jsonl").stat().st_size / 1e6
            print(f"[{_OUT_SUB}] requests.jsonl = {n} lines, {mb:.1f} MB")
            submit_batch(client, bd / "requests.jsonl", RESPONSES_ENDPOINT, bd / "meta.json")
            print(f"[{_OUT_SUB}] submitted a 24h batch — re-run `collect` when it resolves.")
            return 0

    # ---- assemble per-replicate score matrices ---------------------------- #
    out = root / "results" / _OUT_SUB
    out.mkdir(parents=True, exist_ok=True)
    by_b: dict[int, list] = {}
    present = []
    for s in specs:
        rec = cache.get(cache_dir, s.cache_key())
        if not (rec and rec.get("text")):
            continue
        present.append(rec)
        m = s.meta
        for sc in json.loads(rec["text"])["scores"]:
            by_b.setdefault(m["b"], []).append(
                {"letter_id": m["letter_id"], "dimension": sc["dimension"], "score": sc["score"]}
            )

    done = []
    for b, rows in sorted(by_b.items()):
        df = pd.DataFrame(rows)
        wide = df.groupby(["letter_id", "dimension"])["score"].mean().unstack()
        wide.to_parquet(out / f"scores_b{b:03d}.parquet")
        done.append({"b": b, "n_letters": int(wide.shape[0]), "K": int(wide.shape[1])})

    usd = actual_usd(present, cfg.get("pricing"), batch=True)
    (out / "cost_ledger.json").write_text(
        json.dumps(
            {
                "n_specs": len(specs),
                "n_present": len(present),
                "n_replicates": len(done),
                "rep_depth": 1,
                "mode": "batch",
                "reported_basis": "batch",
                "realized_usd_batch": usd,
                "replicates": done,
            },
            indent=2,
        )
    )
    print(
        f"[{_OUT_SUB}] {len(present)}/{len(specs)} present -> {len(done)} replicate score "
        f"matrices | ${usd:.2f} batch"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
