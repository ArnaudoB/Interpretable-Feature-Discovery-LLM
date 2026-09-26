"""Step 5: elicit the pointwise rubrics (feature-creation), for M1/M2 and M3.

    python scripts/05_elicit_rubrics.py full --config config.yaml   # whole-dataset rubric (M1/M2)
    python scripts/05_elicit_rubrics.py reps --config config.yaml --x 50   # E x 40-letter samples
    python scripts/05_elicit_rubrics.py reps --config config.yaml --collect # poll a submitted reps batch

The pointwise baseline elicits its rubric with the evidence-first feature-creation prompt
(POINTWISE_FEATURES_PROMPT, high reasoning):

  * ``full``  -- one call over all 200 letters -> rubric_stability/full_dataset_rubric.json.
                 This whole-dataset rubric is scored once (M1) and cost-matched (M2) in step 7.
  * ``reps``  -- x independent calls, each on a random 40-letter sample ->
                 rubric_stability/reps/rep_XXX.json. Fusing them gives the arms 2a/2b rubric
                 (step 6b) and, by clustering, the M3a/b/c rubrics (step 6).

Cached under ``cache/responses`` (shipped) so re-runs do not re-bill. The ``full`` rubric is
tab:feats-m1.
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
    run_smoke,
    build_jsonl,
    submit_batch,
    poll_until_done,
    collect_into_cache,
)
from core.judging.cache import partition_cached
from core.judging.requests import build_scoring_spec, RESPONSES_ENDPOINT
from core.judging.pricing import actual_usd
from core.openai_client import get_client
import prompts as ELIC

_SYS = "You design an interpretable scoring rubric and output only JSON."


def _root(args):
    return Path(args.config).parent


def _text(cfg, root):
    essays = pd.read_parquet(root / cfg["graph"]["out_dir"] / "essays.parquet")
    return dict(zip(essays["letter_id"], essays["text"]))


def _block(text, ids):
    return "\n\n".join(f"Letter {i + 1}:\n{text[lid]}" for i, lid in enumerate(ids))


def _rep_sample(text, base_seed, r, k):
    rng = np.random.default_rng(
        int(hashlib.md5(f"{base_seed}:stability:{r}".encode()).hexdigest()[:8], 16)
    )
    ids = sorted(text)
    return [ids[i] for i in rng.choice(len(ids), size=k, replace=False)]


def _rep_specs(cfg, text, x, sample_size):
    j = cfg["judge"]
    base = int(j.get("seed", 42))
    specs = []
    for r in range(x):
        sample = _rep_sample(text, base, r, sample_size)
        specs.append(
            build_scoring_spec(
                custom_id=f"rubric_stab:r{r:03d}",
                model=j["reasoning_model"],
                system=_SYS,
                user=ELIC.pointwise_features_prompt(_block(text, sample)),
                json_schema=ELIC.POINTWISE_FEATURES_SCHEMA,
                max_output_tokens=int(j["max_out_rubric"]),
                reasoning_effort=j["high_effort"],
                meta={"rep": r, "sample_size": sample_size, "sample_letter_ids": sample},
            )
        )
    return specs


def cmd_full(cfg, root, client):
    j = cfg["judge"]
    text = _text(cfg, root)
    cache_dir = root / j["cache_dir"]
    spec = build_scoring_spec(
        custom_id="rubric:full_dataset",
        model=j["reasoning_model"],
        system=_SYS,
        user=ELIC.pointwise_features_prompt(_block(text, sorted(text))),
        json_schema=ELIC.POINTWISE_FEATURES_SCHEMA,
        max_output_tokens=int(j["max_out_rubric"]),
        reasoning_effort=j["high_effort"],
    )
    rec = cache.get(cache_dir, spec.cache_key())
    if rec is None or not (rec.get("text") or "").strip():
        [rec] = run_smoke(client, [spec], cache_dir)
    feats = json.loads(rec["text"]).get("features", [])
    out = root / "results" / "rubric_stability"
    out.mkdir(parents=True, exist_ok=True)
    (out / "full_dataset_rubric.json").write_text(
        json.dumps(
            {
                "n_letters": len(text),
                "n_features": len(feats),
                "usage": rec.get("usage", {}),
                "usd_sync": actual_usd([rec], cfg.get("pricing"), batch=False),
                "features": feats,
            },
            indent=2,
        )
    )
    print(
        f"[full] {len(feats)} features over {len(text)} letters -> rubric_stability/full_dataset_rubric.json"
    )


def _write_reps(root, specs, recs):
    reps_dir = root / "results" / "rubric_stability" / "reps"
    reps_dir.mkdir(parents=True, exist_ok=True)
    present = []
    for s in specs:
        rec = recs.get(s.custom_id)
        if not (rec and (rec.get("text") or "").strip()):
            continue
        feats = json.loads(rec["text"]).get("features", [])
        m = s.meta
        (reps_dir / f"rep_{m['rep']:03d}.json").write_text(
            json.dumps(
                {
                    "rep": m["rep"],
                    "sample_size": m["sample_size"],
                    "sample_letter_ids": m["sample_letter_ids"],
                    "model": s.model,
                    "n_features": len(feats),
                    "features": feats,
                    "usage": rec.get("usage", {}),
                },
                indent=2,
            )
        )
        present.append(rec)
    return present


def cmd_reps(cfg, root, client, x, sample_size, collect):
    j = cfg["judge"]
    text = _text(cfg, root)
    cache_dir = root / j["cache_dir"]
    specs = _rep_specs(cfg, text, x, sample_size)
    bd = root / "batch" / "rubric_stability"
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
    cached, todo = partition_cached(cache_dir, specs)
    print(f"[reps] x={x} sample_size={sample_size} | cached {len(cached)} | to run {len(todo)}")
    if todo and not collect:
        if x <= 50:
            run_smoke(client, todo, cache_dir)  # realtime, cached
        else:
            bd.mkdir(parents=True, exist_ok=True)
            build_jsonl(todo, bd / "requests.jsonl")
            submit_batch(client, bd / "requests.jsonl", RESPONSES_ENDPOINT, bd / "meta.json")
            print("[reps] submitted a 24h batch — re-run with --collect when it resolves.")
            return
    recs = {s.custom_id: cache.get(cache_dir, s.cache_key()) for s in specs}
    present = _write_reps(root, specs, recs)
    print(f"[reps] wrote {len(present)}/{len(specs)} rep_*.json under rubric_stability/reps/")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=["full", "reps"])
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--x", type=int, default=50, help="reps: number of 40-letter elicitations")
    p.add_argument("--sample-size", type=int, default=40)
    p.add_argument("--collect", action="store_true", help="reps: poll a previously submitted batch")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    client = get_client()
    if args.command == "full":
        cmd_full(cfg, _root(args), client)
    else:
        cmd_reps(cfg, _root(args), client, args.x, args.sample_size, args.collect)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
