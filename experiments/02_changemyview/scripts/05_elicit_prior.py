"""Step 5: the prior-only panel P -- 16 criteria named with no data shown.

    python scripts/05_elicit_prior.py rederive --out-dir DIR     # offline (see below)
    python scripts/05_elicit_prior.py collect  --out-dir DIR     # replay from the cache
    python scripts/05_elicit_prior.py run --in-place --yes       # the (billed, realtime) draws

gpt-5.4 (high reasoning, sync) is asked to name exactly 16 dimensions for assessing how
persuasive a reply to someone's stated view is. It sees no exchanges, no outcomes and no
corpus description (prompt: ``prompts/prior.py``). ``prior.draws`` (5) independent calls are
made -- ``cache_version="draw<d>"`` keeps their cache keys apart -- and **draw 0 is promoted by
position** to ``features_blind.json``, fixed before anything is scored, so no draw is selected
on outcome. That panel is what the P + BT and P + PW cells score.

This is the blind variant (``named=False``). The open-count elicitation of the sibling cell
``feature_prior_root_open`` uses ``prompts/prior_open.py`` and is not driven by this step; its
shipped panels are read as data by step 14.

**The draws' responses are not in the cache.** ``feature_prior_root/cache/responses/`` holds no
response records, so ``collect`` misses. ``rederive`` is the offline route: it takes the parsed panels the run saved in
``panels_blind.json`` and re-derives the promoted ``features_blind.json`` from them.

Writes <prior cell>/results/panels_blind.json and <prior cell>/features_blind.json.
"""

from __future__ import annotations

import argparse
import json
import logging

import _paths  # noqa: F401
import _config as C
import _judge as J
from core.judging import cache
from core.judging.batch import realtime_call
from core.judging.pricing import actual_usd
from core.judging.requests import build_scoring_spec


def settings():
    return C.pipeline()["prior"]


def build_specs():
    """One spec per draw, exactly as the run built them."""
    s, P = settings(), C.prompt("prior")
    system, user = P.prior_messages(n=P.N_FEATURES, named=False)
    return [
        build_scoring_spec(
            f"prior:{s['variant']}:{d}",
            s["model"],
            system,
            user,
            P.PRIOR_SCHEMA,
            int(s["max_out"]),
            reasoning_effort=s["effort"],
            cache_version=f"draw{d}",
        )
        for d in range(int(s["draws"]))
    ]


def write(panels, usages, cell_dir, out_dir):
    s, P = settings(), C.prompt("prior")
    tag = s["variant"]
    res = C.work_path(cell_dir / "results", out_dir)
    res.mkdir(parents=True, exist_ok=True)
    (res / f"panels_{tag}.json").write_text(
        json.dumps(
            [
                {"draw": i, "criteria": p, "usage": u}
                for i, (p, u) in enumerate(zip(panels, usages))
            ],
            indent=2,
        )
    )
    src = (
        f"prior-only elicitation, {s['model']} effort={s['effort']}, variant={tag}, draw 0 of "
        f"{len(panels)}, n={P.N_FEATURES}; no exchanges/outcomes/corpus shown "
        f"(prompts/prior.py)"
    )
    fj = P.to_features_json(panels[0], src)
    (C.work_path(cell_dir, out_dir) / f"features_{tag}.json").write_text(json.dumps(fj, indent=2))
    print(
        f"[prior] {len(panels)} draws, counts {[len(p) for p in panels]}; draw 0 promoted -> "
        f"{C.work_path(cell_dir, out_dir) / f'features_{tag}.json'}"
    )
    sets = [{c["name"].strip().lower() for c in p} for p in panels]
    for i in range(1, len(sets)):
        print(f"  draw 0 vs {i}: {len(sets[0] & sets[i])} shared names of {len(sets[0])}")


def main() -> int:
    ap = J.add_stage_args(
        argparse.ArgumentParser(description=__doc__.splitlines()[0]),
        stages=("run", "collect", "rederive"),
    )
    a = ap.parse_args()
    J.guard(a.stage, a.yes)
    C.check_writable(a.out_dir, a.in_place)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    s, P = settings(), C.prompt("prior")
    cell_dir, cache_dir = C.cell(s["cell"]), C.cell_cache(s["cell"])
    specs = build_specs()

    if a.stage == "rederive":
        saved = json.loads(
            C.input_path(
                cell_dir / "results" / f"panels_{s['variant']}.json", a.out_dir
            ).read_text()
        )
        print(
            f"[prior] rederive: panels taken from the saved panels_{s['variant']}.json, "
            f"not from the cache"
        )
        panels = [d["criteria"] for d in saved]
        usages = [d.get("usage") for d in saved]
    else:
        if a.stage == "run":
            for sp in specs:
                rec = cache.get(cache_dir, sp.cache_key())
                if rec is None or not (rec.get("text") or "").strip():
                    rec = realtime_call(J.client(), sp)
                    if (rec.get("text") or "").strip():
                        cache.put(cache_dir, sp.cache_key(), rec)
        recs = J.replay(specs, cache_dir, label="prior")
        panels = [P.parse_prior(recs[sp.custom_id]["text"], n=P.N_FEATURES) for sp in specs]
        usages = [recs[sp.custom_id].get("usage") for sp in specs]
        print(
            f"[prior] {len(specs)} sync call(s), ${actual_usd(list(recs.values()), batch=False):.3f}"
        )
    write(panels, usages, cell_dir, a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
