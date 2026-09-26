"""Step 2: run the pairwise comparative judge over all 4000 pairs (Batch API).

    python scripts/02_judge_comparative.py smoke   --config config.yaml   # measure tokens/cost
    python scripts/02_judge_comparative.py submit  --config config.yaml   # launch the 24h batch
    python scripts/02_judge_comparative.py collect --config config.yaml   # poll + assemble artifacts

Each pair is sent to the pairwise judge (``judge.comparative_model``, no reasoning) with the
COMPARATIVE prompt; the judge names the dimensions on which the two letters differ and which
letter shows more of each. ``collect`` writes:

  * results/comparative_judgments.parquet  -- per-pair raw differences
  * results/phrases.parquet                -- one row per (pair, cited dimension) with winner
  * results/pairs.npy, pair_order.json, essay_index.json  -- letter-index design inputs for the fit

Responses are content-addressed and cached under ``cache/responses`` (shipped), so re-running
never re-bills already-collected pairs. First stage of the Ours row of tab:main.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import _paths  # noqa: F401  (sys.path bootstrap; see scripts/_paths.py)
from core.canonical import normalize_quality
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
from core.judging.pricing import price_for, BATCH_DISCOUNT
from core.openai_client import get_client
import prompts as ELIC


def _root(args):
    return Path(args.config).parent


def _load_text(cfg, root):
    essays = pd.read_parquet(root / cfg["graph"]["out_dir"] / "essays.parquet")
    return dict(zip(essays["letter_id"], essays["text"]))


def _comparative_specs(cfg, text, pairs, effort):
    j = cfg["judge"]
    specs = []
    for _, r in pairs.iterrows():
        first = r["shown_first"]
        other = r["essay_b_id"] if first == r["essay_a_id"] else r["essay_a_id"]
        sysm, usr = ELIC.comparative_messages(text[first], text[other])
        specs.append(
            build_scoring_spec(
                custom_id=f"cmp:{int(r['pair_index']):06d}",
                model=j["comparative_model"],
                system=sysm,
                user=usr,
                json_schema=ELIC.COMPARATIVE_SCHEMA,
                max_output_tokens=int(j["max_out_comparative"]),
                reasoning_effort=effort,
                meta={"pair_index": int(r["pair_index"]), "shown_first": first, "other": other},
            )
        )
    return specs


def cmd_smoke(cfg, root, client):
    j = cfg["judge"]
    text = _load_text(cfg, root)
    pairs = pd.read_parquet(root / cfg["graph"]["out_dir"] / "pairs.parquet")
    cache_dir = root / j["cache_dir"]
    specs = _comparative_specs(cfg, text, pairs.iloc[: int(j["smoke_pairs"])], j["judge_effort"])
    recs = run_smoke(client, specs, cache_dir)
    cin = np.mean([r["usage"].get("input_tokens", 0) for r in recs])
    cout = np.mean([r["usage"].get("output_tokens", 0) for r in recs])
    ndiff = np.mean([len(json.loads(r["text"]).get("differences", [])) for r in recs])
    pin, pout = price_for(j["comparative_model"])
    bulk = len(pairs) * (cin * pin + cout * pout) / 1e6 * BATCH_DISCOUNT
    print(f"[smoke] comparative: n={len(recs)} in~{cin:.0f} out~{cout:.0f} avg_dims~{ndiff:.1f}")
    print(f"[smoke] projected bulk over {len(pairs)} pairs (batch): ${bulk:.2f}")
    (root / "results").mkdir(exist_ok=True)
    (root / "results" / "smoke_report.json").write_text(
        json.dumps(
            {
                "comparative": {
                    "n_pairs": int(len(pairs)),
                    "in": float(cin),
                    "out": float(cout),
                    "avg_dims": float(ndiff),
                    "bulk_usd": bulk,
                }
            },
            indent=2,
        )
    )


def cmd_submit(cfg, root, client):
    j = cfg["judge"]
    text = _load_text(cfg, root)
    pairs = pd.read_parquet(root / cfg["graph"]["out_dir"] / "pairs.parquet")
    specs = _comparative_specs(cfg, text, pairs, j["judge_effort"])
    cache_dir = root / j["cache_dir"]
    cached, todo = partition_cached(cache_dir, specs)
    bd = root / "batch" / "comparative"
    bd.mkdir(parents=True, exist_ok=True)
    print(f"[submit] {len(specs)} specs | cached {len(cached)} | to submit {len(todo)}")
    if not todo:
        (bd / "meta.json").write_text(json.dumps({"batch_id": None, "status": "all_cached"}))
        return
    build_jsonl(todo, bd / "requests.jsonl")
    submit_batch(client, bd / "requests.jsonl", RESPONSES_ENDPOINT, bd / "meta.json")
    print("[submit] done — run `collect` to poll + assemble artifacts.")


def cmd_collect(cfg, root, client):
    j = cfg["judge"]
    text = _load_text(cfg, root)
    pairs = pd.read_parquet(root / cfg["graph"]["out_dir"] / "pairs.parquet")
    specs = _comparative_specs(cfg, text, pairs, j["judge_effort"])
    cache_dir = root / j["cache_dir"]
    bd = root / "batch" / "comparative"
    meta = (
        json.loads((bd / "meta.json").read_text())
        if (bd / "meta.json").exists()
        else {"batch_id": None}
    )
    if meta.get("batch_id"):
        batch = poll_until_done(client, meta["batch_id"], interval_s=60, meta_path=bd / "meta.json")
        collect_into_cache(client, batch, {s.custom_id: s for s in specs}, cache_dir, bd)

    res = root / "results"
    res.mkdir(exist_ok=True)
    letter_ids = sorted(text)
    eidx = {lid: i for i, lid in enumerate(letter_ids)}
    recs = {s.custom_id: cache.get(cache_dir, s.cache_key()) for s in specs}
    ph, jrows, prows, order = [], [], [], []
    for s in specs:
        r = recs[s.custom_id]
        if not (r and r.get("text")):
            continue
        m = s.meta
        pid = str(m["pair_index"])
        first = m["shown_first"]
        other = m["other"]
        diffs = json.loads(r["text"])["differences"]
        order.append(pid)
        prows.append((eidx[first], eidx[other]))
        jrows.append(
            {
                "pair_id": pid,
                "shown_first": first,
                "other": other,
                "n_dims": len(diffs),
                "differences": json.dumps(diffs),
            }
        )
        for d in diffs:
            ph.append(
                {
                    "pair_id": pid,
                    "phrase": d["dimension"],
                    "description": d.get("description", ""),
                    "phrase_norm": normalize_quality(d["dimension"]),
                    "winner_slot": d["winner"],
                }
            )
    pd.DataFrame(ph).to_parquet(res / "phrases.parquet", index=False)
    pd.DataFrame(jrows).to_parquet(res / "comparative_judgments.parquet", index=False)
    np.save(res / "pairs.npy", np.array(prows, dtype=int))
    (res / "pair_order.json").write_text(json.dumps(order))
    (res / "essay_index.json").write_text(json.dumps(eidx))
    print(
        f"[collect] {len(order)} pairs, {len(ph)} dimension rows, "
        f"{pd.DataFrame(ph)['phrase_norm'].nunique()} unique dimensions -> phrases.parquet"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=["smoke", "submit", "collect"])
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    client = get_client()
    {"smoke": cmd_smoke, "submit": cmd_submit, "collect": cmd_collect}[args.command](
        cfg, _root(args), client
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
