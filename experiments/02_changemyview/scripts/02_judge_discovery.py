"""Step 2: the comparative discovery judge -- 6,000 pairwise comparisons, no verdict asked.

    python scripts/02_judge_discovery.py collect --out-dir DIR          # offline replay
    python scripts/02_judge_discovery.py smoke   --out-dir DIR --yes    # 8 pairs, real cost
    python scripts/02_judge_discovery.py submit  --in-place --yes       # the full batch

For each edge of step 1's 40-regular graph, gpt-5.4-mini (no reasoning) sees Exchange A and
Exchange B -- A being the graph's frozen ``shown_first`` -- and lists the dimensions on which
they differ and which exchange shows more of each; it names no overall winner (prompt:
``prompts/discovery.py``). ``collect`` explodes the responses into one row per cited dimension,
each phrase normalized for the taxonomy step, and writes the discovery fit's inputs.

The labelled-discovery ``reveal_delta`` option is used by no cell the paper cites and is not
implemented.

Writes <discovery cell>/results/{phrases.parquet, comparative_judgments.parquet, pairs.npy,
pair_order.json, essay_index.json} (and smoke_report.json from ``smoke``).
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

import _paths  # noqa: F401
import _config as C
import _judge as J
from core.canonical import normalize_quality
from core.judging.pricing import BATCH_DISCOUNT, price_for
from core.judging.requests import build_scoring_spec

CELL = "discovery"


def load(out_dir=None):
    """``(cfg, text by item_id, pairs)`` -- step 0/1 outputs, regenerated copies preferred."""
    cfg = C.cell_config(CELL)
    g = C.source_path(cfg["graph"]["out_dir"])
    essays = pd.read_parquet(C.input_path(g / "essays.parquet", out_dir))
    pairs = pd.read_parquet(C.input_path(g / "pairs.parquet", out_dir))
    return cfg, dict(zip(essays["item_id"], essays["text"])), pairs


def build_specs(cfg, text, pairs):
    """One Responses-API spec per comparison, exactly as the run built them."""
    P = C.prompt("discovery")
    j = cfg["judge"]
    if j.get("reveal_delta"):
        raise NotImplementedError("judge.reveal_delta is the uncited labelled-discovery variant")
    specs = []
    for r in pairs.itertuples(index=False):
        first = r.shown_first
        other = r.essay_b_id if first == r.essay_a_id else r.essay_a_id
        system, user = P.comparative_messages(text[first], text[other])
        specs.append(
            build_scoring_spec(
                custom_id=f"cmp:{int(r.pair_index):06d}",
                model=j["comparative_model"],
                system=system,
                user=user,
                json_schema=P.COMPARATIVE_SCHEMA,
                max_output_tokens=int(j["max_out_comparative"]),
                reasoning_effort=j["judge_effort"],
                meta={"pair_index": int(r.pair_index), "shown_first": first, "other": other},
            )
        )
    return specs


def assemble(specs, recs, text, res):
    """Cited dimensions -> the fit's inputs. A response that fails to parse is skipped."""
    item_ids = sorted(text)
    eidx = {l: i for i, l in enumerate(item_ids)}
    ph, jrows, prows, order = [], [], [], []
    n_empty = n_bad = 0
    for s in specs:
        r = recs.get(s.custom_id)
        if not (r and r.get("text")):
            n_empty += 1
            continue
        try:  # a rare response truncates at the token cap
            diffs = json.loads(r["text"])["differences"]
        except (json.JSONDecodeError, KeyError):
            n_bad += 1
            continue
        m = s.meta
        pid, first, other = str(m["pair_index"]), m["shown_first"], m["other"]
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
    res.mkdir(parents=True, exist_ok=True)
    ph = pd.DataFrame(ph)
    ph.to_parquet(res / "phrases.parquet", index=False)
    pd.DataFrame(jrows).to_parquet(res / "comparative_judgments.parquet", index=False)
    np.save(res / "pairs.npy", np.array(prows, dtype=int))
    (res / "pair_order.json").write_text(json.dumps(order))
    (res / "essay_index.json").write_text(json.dumps(eidx))
    print(
        f"[assemble] {len(order)}/{len(specs)} pairs used ({n_empty} empty, {n_bad} "
        f"malformed), {len(ph)} cited dimensions, {ph['phrase_norm'].nunique()} unique "
        f"-> {res}"
    )


def main() -> int:
    ap = J.add_stage_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    a = ap.parse_args()
    J.guard(a.stage, a.yes)
    C.check_writable(a.out_dir, a.in_place)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg, text, pairs = load(a.out_dir)
    specs = build_specs(cfg, text, pairs)
    cache_dir = C.cell_cache(CELL)
    res = C.work_path(C.cell(CELL) / "results", a.out_dir)
    batch_dir = C.work_path(C.cell(CELL) / "batch" / "comparative", a.out_dir)
    j = cfg["judge"]

    if a.stage == "smoke":
        recs = J.smoke(specs[: int(j["smoke_pairs"])], cache_dir)
        c_in = float(np.mean([r["usage"].get("input_tokens", 0) for r in recs]))
        c_out = float(np.mean([r["usage"].get("output_tokens", 0) for r in recs]))
        ndiff = float(np.mean([len(json.loads(r["text"]).get("differences", [])) for r in recs]))
        pin, pout = price_for(j["comparative_model"])
        bulk = len(specs) * (c_in * pin + c_out * pout) / 1e6 * BATCH_DISCOUNT
        print(
            f"[smoke] n={len(recs)} in~{c_in:.0f} out~{c_out:.0f} avg_dims~{ndiff:.1f} -> "
            f"{len(specs)} pairs ~ ${bulk:.2f} batch"
        )
        res.mkdir(parents=True, exist_ok=True)
        (res / "smoke_report.json").write_text(
            json.dumps(
                {
                    "comparative": {
                        "n_pairs": len(specs),
                        "in": c_in,
                        "out": c_out,
                        "avg_dims": ndiff,
                        "bulk_usd": bulk,
                    }
                },
                indent=2,
            )
        )
    elif a.stage == "submit":
        J.submit(specs, cache_dir, batch_dir)
    else:
        J.poll(specs, cache_dir, batch_dir)
        recs = J.replay(specs, cache_dir, allow_missing=a.allow_missing, label=CELL)
        assemble(specs, recs, text, res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
