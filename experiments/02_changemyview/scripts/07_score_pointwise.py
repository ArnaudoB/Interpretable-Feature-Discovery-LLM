"""Step 7: score a criterion panel pointwise -- one 0-10 rating per exchange and criterion.

    python scripts/07_score_pointwise.py collect --out-dir DIR                # every PW cell
    python scripts/07_score_pointwise.py collect --out-dir DIR --cell pw_prior
    python scripts/07_score_pointwise.py audit                                # free: invalid panels
    python scripts/07_score_pointwise.py smoke  --out-dir DIR --yes --cell pw_prior
    python scripts/07_score_pointwise.py submit --in-place --yes --cell pw_prior
    python scripts/07_score_pointwise.py retry  --in-place --yes --cell pw_prior

Every exchange of the cell's item table is rated 0-10 on all K criteria of its ``features.json``
in a single call -- no anchor graph, no comparisons, no Bradley-Terry (prompt:
``prompts/score_pointwise.py``, criterion text byte-identical to the BT prompt's). Holding the
criteria fixed, this is the contrast that asks whether the pairwise apparatus earns its keep.
Scores are written in the BT cells' schema; "anchor" means the training split here.

``retry`` re-calls every exchange whose cached panel is incomplete or malformed (gpt-5.4-mini
sometimes skips a block of criteria and still reports completion): the invalid record is moved
to ``<cache>/incomplete/try<k>/`` for provenance and the identical request re-issued. It
REWRITES the cache, so it is a paid stage behind ``--yes`` and never part of a replay.

Cells: pipeline.yaml ``score_pointwise.cells``. Every shipped spec is cached; the 2,399 of 2,400
exchanges in ``pw_rep_a`` are because one cached response (wa2251, test) is a run of whitespace
that hit the 4,000-token cap and does not parse, so assembly drops it, as the run did.

Writes <cell>/results/{scores_anchor.parquet, scores_test.parquet, fit_report.json}.
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd

import _paths  # noqa: F401
import _config as C
import _judge as J
from core.judging import cache
from core.judging.batch import run_concurrent
from core.judging.pricing import BATCH_DISCOUNT, price_for
from core.judging.requests import build_scoring_spec


def load(name: str, out_dir=None):
    """``(cfg, items, criteria)`` for one pointwise cell."""
    cfg = C.cell_config(name)
    g = C.source_path(cfg["graph"]["out_dir"])
    items = pd.read_parquet(C.input_path(g / "items.parquet", out_dir))
    feats = C.input_path(C.cell(name) / C.source_path(cfg["source"]["features"]).name, out_dir)
    return cfg, items, json.loads(feats.read_text())["criteria"]


def build_specs(cfg, items, criteria):
    """One Responses-API spec per exchange, exactly as the run built them."""
    P = C.prompt("score_pointwise")
    j = cfg["judge"]
    out = []
    for r in items.itertuples(index=False):
        system, user = P.pointwise_messages(r.text, criteria)
        out.append(
            build_scoring_spec(
                custom_id=f"pw:{r.item_id}",
                model=j["comparative_model"],
                system=system,
                user=user,
                json_schema=P.POINTWISE_SCHEMA,
                max_output_tokens=int(j["max_out_comparative"]),
                reasoning_effort=j["judge_effort"],
                meta={"item_id": r.item_id, "split": r.split},
            )
        )
    return out


def validate(rec, names, lo=0, hi=10) -> str | None:
    """``None`` if a cached record is a complete, well-formed panel, else what is wrong.

    Stricter than ``parse_pointwise``, which drops unknown names and leaves a skipped
    criterion as NaN.
    """
    if not (rec and rec.get("text")):
        return "empty"
    t = rec["text"].strip()
    if t.startswith("```"):
        t = t[t.find("{") : t.rfind("}") + 1]
    try:
        got_items = json.loads(t).get("criteria")
    except (json.JSONDecodeError, AttributeError):
        return "malformed"
    if not isinstance(got_items, list):
        return "malformed"
    got = [
        it.get("name", "").strip()
        if isinstance(it, dict) and isinstance(it.get("name"), str)
        else None
        for it in got_items
    ]
    if len(got) != len(set(got)):
        return "duplicate_name"
    if set(got) - set(names):
        return "unknown_name"
    if len(got) < len(names):
        return "missing_criteria"
    for it in got_items:
        sc = it.get("score")
        if isinstance(sc, bool) or not isinstance(sc, int) or not lo <= sc <= hi:
            return "bad_score"
    return None


def assemble(specs, recs, criteria, res):
    P = C.prompt("score_pointwise")
    names = [c["name"] for c in criteria]
    rows, empty, bad = [], 0, 0
    for s in specs:
        rec = recs.get(s.custom_id)
        if not (rec and rec.get("text")):
            empty += 1
            continue
        try:
            sc = P.parse_pointwise(rec["text"], names)
        except (json.JSONDecodeError, KeyError, TypeError):
            bad += 1
            continue
        rows.append(
            {
                "item_id": s.meta["item_id"],
                "split": s.meta["split"],
                **{n: sc.get(n, np.nan) for n in names},
            }
        )
    df = pd.DataFrame(rows)
    A = df[names].to_numpy(float)
    cov = float(np.isfinite(A).mean()) if len(df) else 0.0
    incomplete = int((~np.isfinite(A)).any(axis=1).sum()) if len(df) else 0
    res.mkdir(parents=True, exist_ok=True)
    for split, fname in (("train", "scores_anchor.parquet"), ("test", "scores_test.parquet")):
        df[df.split == split].drop(columns=["split"]).reset_index(drop=True).to_parquet(
            res / fname, index=False
        )
    (res / "fit_report.json").write_text(
        json.dumps(
            {
                "n_items": len(df),
                "n_features": len(names),
                "coverage": cov,
                "empty": empty,
                "malformed": bad,
                "incomplete": incomplete,
            },
            indent=2,
        )
    )
    print(
        f"[assemble] {len(df)}/{len(specs)} exchanges scored ({empty} empty, {bad} "
        f"malformed, {incomplete} incomplete); coverage {100 * cov:.2f}% -> {res}"
    )


def audit(specs, cache_dir, names) -> dict:
    """``{cache_key: (spec, why)}`` for every distinct request whose cached panel is invalid."""
    fails = {}
    for s in {s.cache_key(): s for s in specs}.values():
        why = validate(cache.get(cache_dir, s.cache_key()), names)
        if why:
            fails[s.cache_key()] = (s, why)
    return fails


def run_cell(name: str, a) -> None:
    cfg, items, criteria = load(name, a.out_dir)
    names = [c["name"] for c in criteria]
    specs = build_specs(cfg, items, criteria)
    cache_dir = C.cell_cache(name)
    res = C.work_path(C.cell(name) / "results", a.out_dir)
    batch_dir = C.work_path(C.cell(name) / "batch" / "pointwise", a.out_dir)
    print(f"[{name}] {len(specs)} exchanges x {len(criteria)} criteria")

    if a.stage == "smoke":
        P = C.prompt("score_pointwise")
        recs = J.smoke(specs[: a.n], cache_dir)
        c_in = float(np.mean([r["usage"].get("input_tokens", 0) for r in recs]))
        c_out = float(np.mean([r["usage"].get("output_tokens", 0) for r in recs]))
        A = np.array(
            [[P.parse_pointwise(r["text"], names).get(n, np.nan) for n in names] for r in recs],
            dtype=float,
        )
        pin, pout = price_for(cfg["judge"]["comparative_model"])
        sync = len(specs) * (c_in * pin + c_out * pout) / 1e6
        print(
            f"[smoke] n={len(recs)} in~{c_in:.0f} out~{c_out:.0f}; coverage "
            f"{100 * np.isfinite(A).mean():.1f}%; share of ratings in 4-7 "
            f"{np.nanmean((A >= 4) & (A <= 7)):.2f}; ${sync:.2f} sync "
            f"(${sync * BATCH_DISCOUNT:.2f} batch)"
        )
    elif a.stage == "submit":
        J.submit(specs, cache_dir, batch_dir)
    elif a.stage == "audit":
        fails = audit(specs, cache_dir, names)
        print(
            f"[audit {name}] {len({s.cache_key() for s in specs})} distinct requests | invalid "
            f"{len(fails)}: {pd.Series([w for _, w in fails.values()]).value_counts().to_dict()}"
        )
    elif a.stage == "retry":
        fails = audit(specs, cache_dir, names)
        for k in range(1, a.max_tries + 1):
            if not fails:
                break
            arch = cache_dir / "incomplete" / f"try{k}"
            arch.mkdir(parents=True, exist_ok=True)
            for h in fails:
                p = cache.cache_path(cache_dir, h)
                if p.exists():
                    p.rename(arch / p.name)
            todo = [s for s, _ in fails.values()]
            t0 = time.time()
            _, errors = run_concurrent(
                J.client(), todo, cache_dir, max_workers=int(cfg["judge"]["sync_workers"])
            )
            fails = audit(specs, cache_dir, names)
            print(
                f"[retry {name}] try {k}: called {len(todo)} ({len(errors)} API errors) in "
                f"{time.time() - t0:.0f}s -> still invalid {len(fails)}"
            )
        recs = J.replay(specs, cache_dir, allow_missing=True, label=name)
        assemble(specs, recs, criteria, res)
    else:
        J.poll(specs, cache_dir, batch_dir)
        recs = J.replay(specs, cache_dir, allow_missing=a.allow_missing, label=name)
        assemble(specs, recs, criteria, res)


def main() -> int:
    ap = J.add_stage_args(
        argparse.ArgumentParser(description=__doc__.splitlines()[0]),
        stages=("smoke", "submit", "collect", "audit", "retry"),
    )
    ap.add_argument(
        "--cell",
        action="append",
        default=None,
        help="cell(s) to run (default: pipeline.yaml score_pointwise.cells)",
    )
    ap.add_argument("-n", type=int, default=8, help="smoke: number of exchanges")
    ap.add_argument("--max-tries", type=int, default=3, help="retry: re-call passes")
    a = ap.parse_args()
    J.guard(a.stage, a.yes)
    if a.stage != "audit":
        C.check_writable(a.out_dir, a.in_place)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in a.cell or C.pipeline()["score_pointwise"]["cells"]:
        run_cell(name, a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
