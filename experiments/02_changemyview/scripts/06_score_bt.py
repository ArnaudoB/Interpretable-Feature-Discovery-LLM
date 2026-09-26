"""Step 6: score a criterion panel by Bradley-Terry -- 32,000 comparisons per cell, then the fit.

    python scripts/06_score_bt.py collect --out-dir DIR                 # every BT cell, offline
    python scripts/06_score_bt.py collect --out-dir DIR --cell bt_prior
    python scripts/06_score_bt.py smoke   --out-dir DIR --yes           # 8 anchor pairs
    python scripts/06_score_bt.py submit  --in-place --yes --cell bt_prior

Judging: every edge of the cell's step-1 graph -- the 6,400
anchor comparisons and the 25,600 test->anchor ones -- is judged in ONE call on all K criteria
of the cell's ``features.json``, a forced A/B winner per criterion (prompt:
``prompts/score_bt.py``). ``collect`` explodes each response into one row per (pair, criterion).

Fit: per criterion, a plain BT with a position term on the anchor
graph gives the 800 anchor exchanges' scores and beta_k (``core.cmv.bt.fit_feature_bt``); each
of the 1,600 test exchanges is then scored alone against its 16 FROZEN anchors
(``core.cmv.bt.score_query_frozen``), so no test outcome ever informs an anchor score. The
"anchor"/"test" split is the task's train/heldout split.

Cells: pipeline.yaml ``score_bt.cells``. ``bt_dropped`` re-uses the flagship graph unchanged.
The supervised ``reveal_delta_scoring`` option is used by no cell the paper cites and is not
implemented.

Writes <cell>/results/{judgments.parquet, scores_anchor.parquet, scores_test.parquet,
beta.json, fit_report.json} (and smoke_report.json from ``smoke``).
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
from core.cmv.bt import fit_feature_bt, score_query_frozen
from core.judging.pricing import BATCH_DISCOUNT, price_for
from core.judging.requests import build_scoring_spec

PAIR_COLS = ["pair_index", "essay_a_id", "essay_b_id", "shown_first", "group"]


def load(name: str, out_dir=None):
    """``(cfg, items, pairs, criteria)`` for one BT cell."""
    cfg = C.cell_config(name)
    if cfg["judge"].get("reveal_delta_scoring"):
        raise NotImplementedError("judge.reveal_delta_scoring is the uncited supervised variant")
    g = C.source_path(cfg["graph"]["out_dir"])
    items = pd.read_parquet(C.input_path(g / "items.parquet", out_dir))
    anchor = pd.read_parquet(C.input_path(g / "anchor_pairs.parquet", out_dir))
    test = pd.read_parquet(C.input_path(g / "test_anchor_pairs.parquet", out_dir))
    pairs = pd.concat([anchor[PAIR_COLS], test[PAIR_COLS]], ignore_index=True)
    feats = C.input_path(C.cell(name) / C.source_path(cfg["source"]["features"]).name, out_dir)
    criteria = json.loads(feats.read_text())["criteria"]
    return cfg, items, pairs, criteria


def build_specs(cfg, items, pairs, criteria):
    """One Responses-API spec per comparison, exactly as the run built them."""
    P = C.prompt("score_bt")
    j = cfg["judge"]
    text = dict(zip(items["item_id"], items["text"]))
    specs = []
    for r in pairs.itertuples(index=False):
        first = r.shown_first
        other = r.essay_b_id if first == r.essay_a_id else r.essay_a_id
        system, user = P.specified_messages(text[first], text[other], criteria)
        specs.append(
            build_scoring_spec(
                custom_id=f"{r.group}:{int(r.pair_index):05d}",
                model=j["comparative_model"],
                system=system,
                user=user,
                json_schema=P.SPECIFIED_SCHEMA,
                max_output_tokens=int(j["max_out_comparative"]),
                reasoning_effort=j["judge_effort"],
                meta={
                    "pair_index": int(r.pair_index),
                    "group": r.group,
                    "shown_first": first,
                    "other": other,
                },
            )
        )
    return specs


def assemble(specs, recs, criteria) -> pd.DataFrame:
    """Responses -> one row per (comparison, criterion) with the named winner."""
    P = C.prompt("score_bt")
    names = [c["name"] for c in criteria]
    rows, empty, bad = [], 0, 0
    for s in specs:
        r = recs.get(s.custom_id)
        if not (r and r.get("text")):
            empty += 1
            continue
        try:
            w = P.parse_specified(r["text"], names)
        except (json.JSONDecodeError, KeyError):
            bad += 1
            continue
        m = s.meta
        for f, win in w.items():
            rows.append(
                {
                    "pair_index": m["pair_index"],
                    "group": m["group"],
                    "feature": f,
                    "shown_first": m["shown_first"],
                    "other": m["other"],
                    "winner": win,
                }
            )
    judg = pd.DataFrame(rows)
    print(
        f"[assemble] {len(specs) - empty - bad}/{len(specs)} comparisons parsed ({empty} "
        f"empty, {bad} malformed); {len(judg)} criterion rows"
    )
    return judg


def fit(cfg, items, judg, criteria, res):
    """Anchor BT per criterion, then frozen scoring of every test exchange."""
    m = cfg["model"]
    names = [c["name"] for c in criteria]
    ridge, tol, mi = float(m["ridge"]), float(m["tol"]), int(m["max_iter"])
    split_map = dict(zip(items["item_id"], items["split"]))
    train_ids = items[items["split"] == "train"]["item_id"].tolist()
    test_ids = items[items["split"] == "test"]["item_id"].tolist()
    aidx = {iid: i for i, iid in enumerate(train_ids)}
    nA = len(train_ids)

    anchor = judg[judg["group"] == "anchor"]
    S_anchor = np.full((nA, len(names)), np.nan)
    beta, conv = {}, []
    for k, f in enumerate(names):
        jk = anchor[anchor["feature"] == f]
        pairs = np.array([[aidx[a], aidx[b]] for a, b in zip(jk["shown_first"], jk["other"])])
        w = (jk["winner"].to_numpy() == "A").astype(float)
        s, bk = fit_feature_bt(pairs, w, nA, ridge=ridge, tol=tol, max_iter=mi)
        S_anchor[:, k] = s
        beta[f] = bk
        conv.append(
            {
                "feature": f,
                "n_pairs": len(jk),
                "beta": round(bk, 3),
                "win_rate_A": round(float(w.mean()), 3),
                "score_sd": round(float(s.std()), 3),
            }
        )
    sA = pd.DataFrame(S_anchor, index=train_ids, columns=names)
    sA.reset_index(names="item_id").to_parquet(res / "scores_anchor.parquet", index=False)

    tj = judg[judg["group"] == "test"].copy()
    tj["q_is_first"] = tj["shown_first"].map(split_map).eq("test")
    tj["q_id"] = np.where(tj["q_is_first"], tj["shown_first"], tj["other"])
    tj["anchor_id"] = np.where(tj["q_is_first"], tj["other"], tj["shown_first"])
    tj["sign"] = np.where(tj["q_is_first"], 1.0, -1.0)
    tj["w"] = (tj["winner"].to_numpy() == "A").astype(float)
    long = sA.reset_index(names="item_id").melt(
        "item_id", var_name="feature", value_name="s_anchor"
    )
    tj = tj.merge(
        long, left_on=["anchor_id", "feature"], right_on=["item_id", "feature"], how="left"
    )
    S_test = np.full((len(test_ids), len(names)), np.nan)
    tpos = {iid: i for i, iid in enumerate(test_ids)}
    kpos = {f: k for k, f in enumerate(names)}
    for (qid, f), grp in tj.groupby(["q_id", "feature"]):
        S_test[tpos[qid], kpos[f]] = score_query_frozen(
            grp["sign"].to_numpy(),
            grp["s_anchor"].to_numpy(),
            grp["w"].to_numpy(),
            beta[f],
            ridge=ridge,
        )
    sT = pd.DataFrame(S_test, index=test_ids, columns=names)
    sT.reset_index(names="item_id").to_parquet(res / "scores_test.parquet", index=False)

    (res / "beta.json").write_text(json.dumps(beta, indent=2))
    (res / "fit_report.json").write_text(
        json.dumps(
            {
                "n_anchor": nA,
                "n_test": len(test_ids),
                "n_features": len(names),
                "test_nan": int(np.isnan(S_test).sum()),
                "per_feature": conv,
            },
            indent=2,
        )
    )
    print(
        f"[fit] anchors {nA} x {len(names)} | test {len(test_ids)} x {len(names)} frozen "
        f"({int(np.isnan(S_test).sum())} NaN) | median |beta| "
        f"{np.median([abs(b) for b in beta.values()]):.3f} -> {res}"
    )


def run_cell(name: str, a) -> None:
    cfg, items, pairs, criteria = load(name, a.out_dir)
    specs = build_specs(cfg, items, pairs, criteria)
    cache_dir = C.cell_cache(name)
    res = C.work_path(C.cell(name) / "results", a.out_dir)
    batch_dir = C.work_path(C.cell(name) / "batch" / "specified", a.out_dir)
    j = cfg["judge"]
    print(f"[{name}] {len(specs)} comparisons x {len(criteria)} criteria")

    if a.stage == "smoke":
        P = C.prompt("score_bt")
        names = [c["name"] for c in criteria]
        smoke = [s for s in specs if s.meta["group"] == "anchor"][: int(j["smoke_pairs"])]
        recs = J.smoke(smoke, cache_dir)
        c_in = float(np.mean([r["usage"].get("input_tokens", 0) for r in recs]))
        c_out = float(np.mean([r["usage"].get("output_tokens", 0) for r in recs]))
        # HALO: per-pair winner agreement across criteria (1.0 = one exchange wins them all)
        agree = [
            max(
                np.mean([v == "A" for v in w.values()]), 1 - np.mean([v == "A" for v in w.values()])
            )
            for w in (P.parse_specified(r["text"], names) for r in recs)
            if w
        ]
        pin, pout = price_for(j["comparative_model"])
        bulk = len(specs) * (c_in * pin + c_out * pout) / 1e6 * BATCH_DISCOUNT
        print(
            f"[smoke] n={len(recs)} in~{c_in:.0f} out~{c_out:.0f} -> ${bulk:.2f} batch; "
            f"HALO median agreement {np.median(agree):.2f}"
        )
        res.mkdir(parents=True, exist_ok=True)
        (res / "smoke_report.json").write_text(
            json.dumps(
                {
                    "n_pairs": len(specs),
                    "in": c_in,
                    "out": c_out,
                    "bulk_usd_batch": bulk,
                    "halo_median_agreement": float(np.median(agree)),
                },
                indent=2,
            )
        )
    elif a.stage == "submit":
        J.submit(specs, cache_dir, batch_dir)
    else:
        J.poll(specs, cache_dir, batch_dir)
        recs = J.replay(specs, cache_dir, allow_missing=a.allow_missing, label=name)
        judg = assemble(specs, recs, criteria)
        res.mkdir(parents=True, exist_ok=True)
        judg.to_parquet(res / "judgments.parquet", index=False)
        if not a.no_fit:
            fit(cfg, items, judg, criteria, res)


def main() -> int:
    ap = J.add_stage_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    ap.add_argument(
        "--cell",
        action="append",
        default=None,
        help="cell(s) to run (default: pipeline.yaml score_bt.cells)",
    )
    ap.add_argument("--no-fit", action="store_true", help="collect: judgments only, no BT fit")
    a = ap.parse_args()
    J.guard(a.stage, a.yes)
    C.check_writable(a.out_dir, a.in_place)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in a.cell or C.pipeline()["score_bt"]["cells"]:
        run_cell(name, a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
