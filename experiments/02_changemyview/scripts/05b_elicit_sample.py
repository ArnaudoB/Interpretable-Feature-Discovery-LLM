"""Step 5b: the data-informed panel S -- criteria named after reading 752 blind exchanges.

    python scripts/05b_elicit_sample.py collect --out-dir DIR     # offline replay
    python scripts/05b_elicit_sample.py dry-run                   # render and price, no call
    python scripts/05b_elicit_sample.py run --in-place --yes      # the (billed, realtime) draw

The counterpart of step 5 that changes one thing: the elicitor sees data. gpt-5.4 (high
reasoning, one sync call) is shown as many (OP view, reply A, reply B) triples of the flagship's
item table as fit a ``sample.budget_tokens`` (900k est.) budget -- at most one pair per distinct
OP, A/B order shuffled under its own seed, no outcome shown -- and names the dimensions on which
the replies differ (prompt: ``prompts/sample.py``; sampling and packing:
``core.cmv.sample_elicit``). The prompt is asserted to be the frozen template plus the three
text columns alone, so no id, split or delta can have leaked. Draw 0 is promoted by position.

``collect`` also writes the REPEAT panel scored by ``pw_sample_rep``: the same criteria, order
permuted with ``sample.reshuffle_seed`` (7). The judge runs at temperature 0, so only a prompt
that differs -- here, by order -- gives the independent second measurement the noise
correction needs; both facts are asserted.

The run used a 900k-token budget and one draw. The context-ceiling probe that preceded it is
not driven here; its record ships as results/context_probe.json.

Writes <sample cell>/results/{panels_sample.json, sample_manifest.json},
<sample cell>/features_sample.json and <rep cell>/features.json.
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
from core.cmv import sample_elicit as SE
from core.judging import cache
from core.judging.batch import realtime_call
from core.judging.pricing import actual_usd, price_for
from core.judging.requests import build_scoring_spec


def settings():
    return C.pipeline()["sample"]


def items_path(out_dir=None):
    g = C.source_path(C.cell_config(settings()["items_from"])["graph"]["out_dir"])
    return C.input_path(g / "items.parquet", out_dir)


def render(out_dir=None, budget=None):
    """``(system, user, sample, candidate pool)`` at the configured budget."""
    s, P = settings(), C.prompt("sample")
    cand = SE.select_pairs(
        SE.build_pair_blocks(pd.read_parquet(items_path(out_dir))), dedup_op=bool(s["dedup_op"])
    )
    sample = SE.pack_to_budget(cand, float(budget if budget is not None else s["budget_tokens"]))
    system, user = P.sample_messages(SE.render_exchanges(sample))
    assert_blind(user, sample)
    return system, user, sample, cand


def assert_blind(user, sample):
    """The prompt is the frozen template + (op, reply_a, reply_b) alone, and A/B is shuffled.

    A keyword scan for "delta" would not work: OP bodies genuinely contain the word (posters
    edit in "I've awarded the delta to ..."), which every arm's judge sees too.
    """
    P = C.prompt("sample")
    if P.EXCHANGES_SENTINEL in user:
        raise AssertionError("the exchanges sentinel was never replaced")
    if user != P.SAMPLE_USER.replace(P.EXCHANGES_SENTINEL, SE.render_exchanges(sample)):
        raise AssertionError("prompt is not template + (op, reply_a, reply_b) alone")
    for col in ("pair_id", "item_id"):
        for v in list(sample[col])[:50] if col in sample.columns else []:
            if str(v) in user:
                raise AssertionError(f"prompt leaks {col}={v!r}")
    if not 0.35 <= sample.a_is_winner.mean() <= 0.65:
        raise AssertionError("A/B order is not shuffled")


def build_specs(out_dir=None):
    """One spec per draw, exactly as the run built them."""
    s, P = settings(), C.prompt("sample")
    system, user, _, _ = render(out_dir)
    return [
        build_scoring_spec(
            f"sample:{d}",
            s["model"],
            system,
            user,
            P.SAMPLE_SCHEMA,
            int(s["max_out"]),
            reasoning_effort=s["effort"],
            cache_version=f"draw{d}",
        )
        for d in range(int(s["draws"]))
    ]


def emit_rep(features: dict, out_dir):
    """The reshuffled repeat panel: same names and definitions, order permuted."""
    s = settings()
    crit = features["criteria"]
    perm = list(np.random.default_rng(int(s["reshuffle_seed"])).permutation(len(crit)))
    shuffled = [crit[i] for i in perm]
    assert {c["name"] for c in shuffled} == {c["name"] for c in crit}, "names changed"
    assert [c["name"] for c in shuffled] != [c["name"] for c in crit], "permutation is the identity"
    by_name = {c["name"]: c["definition"] for c in crit}
    assert all(c["definition"] == by_name[c["name"]] for c in shuffled), "definitions drifted"
    dest = C.work_path(C.cell(s["rep_cell"]), out_dir) / "features.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(
            {
                "n_features": len(shuffled),
                "source": (
                    f"REPEAT measurement of the data-informed panel, reshuffled order "
                    f"(seed {s['reshuffle_seed']}); same names and definitions as "
                    f"feature_sample_root/features_sample.json"
                ),
                "criteria": shuffled,
            },
            indent=2,
        )
    )
    print(f"[sample] repeat panel ({len(shuffled)} criteria, seed {s['reshuffle_seed']}) -> {dest}")


def main() -> int:
    ap = J.add_stage_args(
        argparse.ArgumentParser(description=__doc__.splitlines()[0]),
        stages=("dry-run", "run", "collect"),
    )
    a = ap.parse_args()
    J.guard(a.stage, a.yes)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    s, P = settings(), C.prompt("sample")
    cell_dir, cache_dir = C.cell(s["cell"]), C.cell_cache(s["cell"])

    system, user, sample, cand = render(a.out_dir)
    rep = SE.sample_report(sample, cand)
    est_in = len(system + user) / SE.CHARS_PER_TOKEN
    print(
        f"[sample] {rep['n_pairs']} pairs of {rep['n_pool']} candidates "
        f"({100 * rep['coverage']:.0f}%), splits {rep['by_split']}; prompt ~{est_in:,.0f} "
        f"tokens (~${est_in * price_for(s['model'])[0] / 1e6:.2f} input per draw); winner is "
        f"REPLY A in {100 * rep['winner_is_a']:.1f}%"
    )
    if a.stage == "dry-run":
        return 0
    C.check_writable(a.out_dir, a.in_place)

    specs = build_specs(a.out_dir)
    if a.stage == "run":
        # At the recorded 900k budget the call was accepted, so no fallback to a smaller
        # budget on a context rejection is implemented.
        for sp in specs:
            rec = cache.get(cache_dir, sp.cache_key())
            if rec is None or not (rec.get("text") or "").strip():
                rec = realtime_call(J.client(), sp)
                if (rec.get("text") or "").strip():
                    cache.put(cache_dir, sp.cache_key(), rec)
    recs = J.replay(specs, cache_dir, label="sample")
    panels = [P.parse_sample(recs[sp.custom_id]["text"]) for sp in specs]
    realized = [
        {"draw": d, "budget_tokens": float(s["budget_tokens"]), **rep} for d in range(len(specs))
    ]

    res = C.work_path(cell_dir / "results", a.out_dir)
    res.mkdir(parents=True, exist_ok=True)
    (res / "panels_sample.json").write_text(
        json.dumps(
            [
                {"draw": i, "criteria": p, "usage": recs[sp.custom_id].get("usage"), "sample": r}
                for i, (p, sp, r) in enumerate(zip(panels, specs, realized))
            ],
            indent=2,
        )
    )
    (res / "sample_manifest.json").write_text(
        json.dumps(
            {
                # the item table's location relative to the experiment directory
                "items": str(items_path().relative_to(C.EXPERIMENT)),
                "presentation_seed": SE.PRESENTATION_SEED,
                "chars_per_token": SE.CHARS_PER_TOKEN,
                "dedup_op": bool(s["dedup_op"]),
                "draws": realized,
                "pair_ids_draw0": sorted(sample.pair_id),
            },
            indent=2,
        )
    )
    src = (
        f"data-informed elicitation, {s['model']} effort={s['effort']}, draw 0 of "
        f"{len(specs)}, {rep['n_pairs']} blind (OP, reply A, reply B) exchanges from the "
        f"pooled 400 train + 800 test pairs, budget {float(s['budget_tokens']):,.0f} est "
        f"tokens; no outcomes shown (prompts/sample.py)"
    )
    fj = P.to_features_json(panels[0], src)
    (C.work_path(cell_dir, a.out_dir) / "features_sample.json").write_text(json.dumps(fj, indent=2))
    print(
        f"[sample] draw 0: {len(panels[0])} criteria; ${actual_usd(list(recs.values()), batch=False):.2f} "
        f"when drawn -> {res}"
    )
    emit_rep(fj, a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
