"""Step 8: the baselines -- Tan's text arms, the zero-shot judge, and text embeddings.

    python scripts/08_baselines.py text --out-dir DIR                      # no LLM
    python scripts/08_baselines.py text --out-dir DIR --cell bt_discovered
    python scripts/08_baselines.py zeroshot collect --out-dir DIR          # offline replay
    python scripts/08_baselines.py zeroshot smoke --out-dir DIR --yes      # 8 pairs, realtime
    python scripts/08_baselines.py zeroshot run --in-place --yes           # all 1,600 calls
    python scripts/08_baselines.py embeddings fit --out-dir DIR --store PATH
    python scripts/08_baselines.py embeddings embed --in-place --yes       # API, ~$2

Every arm is the same paired task as step 9: an L1 logistic regression on the signed pair
difference (the shared seed-42 sign deck, so McNemar is valid across arms), C chosen by
5-fold GroupKFold over original posters, one pass over the 800 heldout pairs.

**text** --
#words, interplay, BOW, POS, interplay+POS and all-Tan (``core.cmv.features`` /
``core.cmv.designs``), plus the cell's fitted BT panel where its scores are shipped, on the
cell's own split (400 or 3,411 anchor pairs). Writes the per-pair predictions step 9 reads, and
every arm's nnz / selected C / accuracy / surviving coefficients. The BOW and POS designs come
from the cell's shipped ``design_cache`` (the spaCy pass costs minutes); ``--rebuild-designs``
recomputes them from text. bt_full ships no BT scores (its 54,576 anchor comparisons are not
included), so its BT arm is skipped and its BT-16 rows in the shipped files stay the only record.

**zeroshot** -- gpt-5.4-mini sees the OP's view and
both replies of each heldout pair and names the one that changed the poster's mind (prompt:
``prompts/zeroshot.py``), in BOTH presentation orders: a seeded coin decides which reply is
REPLY A in the first call, the second mirrors it. Primary metric: order-averaged accuracy (an
order-inconsistent pair scores 0.5), compared to the flagship BT arm by paired bootstrap.

**embeddings** (``embed`` / ``fit``) -- whole-reply
text-embedding-3-large vectors differenced within pair (``emb``, 3,072 dims) and 11 reply-OP
cosine statistics (``emb-sim``), at n400 and n3411. ``embed`` calls the embeddings API and
writes a 1.4 GB vector store; ``fit`` reads it (``--store`` to point at a copy). Step 9 reads
the arm's shipped summaries.

**liblinear.** The original runs did not seed liblinear, so it drew from numpy's global RNG as
it found it; the shipped predictions are one draw. Here that RNG is re-seeded
(``task.liblinear_global_seed``) before EVERY fit, so an arm's result depends neither on the
run nor on which arms were fitted before it (seeding once per driver would make bt_full's
arms move just because its BT arm is absent). A re-run repeats itself; it cannot recover the
published draw. For most arms that costs nothing -- they reproduce exactly -- but all-Tan's
cross-validation is nearly tied between C = 0.03 and C = 31.6 on the 400-pair split, and an
unlucky RNG state selects the latter: 481 heldout pairs instead of the shipped (and published)
510. Fresh seeds 0-7 and 42 all give 510.

Writes <cell>/results/{pair_predictions.parquet, arm_sparsity.parquet, arm_sparsity_coefs.json}
(text); <zeroshot cell>/results/{verdicts.parquet, summary.json}; <embeddings
cell>/results/{summary_n*.parquet, pair_predictions_n*.parquet, emb_sim_winrates_n*.parquet}.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest

import _paths  # noqa: F401
import _config as C
import _judge as J
from core.cmv import embeddings as EMB
from core.cmv import features as WF
from core.cmv import pairtask as PT
from core.cmv.designs import (
    SEED,
    boot_ci,
    bt_design,
    build_arm_designs,
    fit_select,
    matched_pairs_from_split,
)
from core.cmv.data import build_pair_units
from core.judging.pricing import actual_usd
from core.judging.requests import build_scoring_spec

warnings.filterwarnings("ignore")  # sklearn's penalty= deprecation flood

#: arm_sparsity_coefs.json keeps this many surviving features per arm
TOP_N = 50


def split(name: str, out_dir=None):
    """The cell's matched pairs (text + train/test split), built from its own config.

    The config's ``discovery_items`` is a run-layout path; it is resolved here and
    asserted, because ``matched_pairs_from_split`` silently skips a path that does not exist --
    which would drop the force-included discovery pairs and change the split.
    """
    cfg = C.cell_config(name)
    disc = C.input_path(C.source_path(cfg["source"]["discovery_items"]), out_dir)
    assert disc.exists(), f"discovery items missing at {disc}"
    cfg["source"]["discovery_items"] = str(disc)
    mp, _ = matched_pairs_from_split(cfg, C.EXPERIMENT)
    return mp


def fit(X, y, train, groups):
    """``fit_select`` with liblinear's global RNG re-seeded first (see the module docstring)."""
    np.random.seed(int(C.cfg()["task"]["liblinear_global_seed"]))
    return fit_select(X, y, train, groups)


# --------------------------------------------------------------------------- #
# text arms
# --------------------------------------------------------------------------- #
def design_cache(name: str, out_dir, rebuild: bool):
    """A writable design cache, pre-filled from the shipped one unless ``rebuild``."""
    shipped = C.cell(name) / "results" / "design_cache"
    dst = C.work_path(shipped, out_dir)
    if rebuild:
        dst = dst.with_name("design_cache_rebuilt")
        return dst
    if dst != shipped and shipped.exists() and not dst.exists():
        shutil.copytree(shipped, dst)
    return dst


def bt_scores_dir(name: str, out_dir):
    """Where the cell's fitted BT scores are (step 6's output, else shipped), or None."""
    d = C.cell(name) / "results"
    for p in (C.work_path(d, out_dir), d):
        if (p / "scores_anchor.parquet").exists() and (p / "scores_test.parquet").exists():
            return p
    return None


def cmd_text(a) -> int:
    for name, arms in C.pipeline()["baselines"]["text"].items():
        if a.cell and name not in a.cell:
            continue
        mp = split(name, a.out_dir)
        train, groups = mp.train.to_numpy(), mp.op_author.to_numpy()
        res = C.work_path(C.cell(name) / "results", a.out_dir)
        res.mkdir(parents=True, exist_ok=True)
        btd = bt_scores_dir(name, a.out_dir)
        print(
            f"[text {name}] {len(mp)} pairs: train {int(train.sum())}, heldout "
            f"{int((~train).sum())}; BT scores: {btd or 'not shipped -> BT arm skipped'}"
        )
        want_root = set(arms["root_arms"]) | ({"bt"} if btd else set())
        want_sp = set(arms["sparsity_arms"]) - (set() if btd else {"bt"})
        designs, y = build_arm_designs(
            mp,
            want_root | want_sp,
            results_dir=btd,
            cache_dir=design_cache(name, a.out_dir, a.rebuild_designs),
            seed=SEED,
        )
        bt = next((k for k in designs if k.startswith("BT-")), None)

        # ---- per-pair predictions ----
        root_order = (
            ["#words", "interplay"]
            + ([bt] if bt else [])
            + [
                k
                for k in ("BOW", "POS", "interplay+POS", "all-Tan")
                if {
                    "BOW": "bow",
                    "POS": "pos",
                    "interplay+POS": "interplay+pos",
                    "all-Tan": "all-tan",
                }[k]
                in arms["root_arms"]
            ]
        )
        preds = {k: fit(designs[k][0], y, train, groups) for k in root_order}
        te = mp.loc[~mp.train, "pair_id"].to_numpy()
        shown = ([bt] if bt else []) + [k for k in root_order if k != bt]
        pd.DataFrame(
            [
                dict(pair_id=p, arm=k, pred=int(pr), y=int(yy))
                for k in shown
                for p, pr, yy in zip(te, preds[k]["pred"], preds[k]["y_test"])
            ]
        ).to_parquet(res / "pair_predictions.parquet", index=False)
        for k in shown:
            print(
                f"  [root] {k:<14} acc {100 * (preds[k]['pred'] == preds[k]['y_test']).mean():.2f}%"
            )

        # ---- L1 sparsity ----
        rows, coefs = [], {}
        for arm, (X, names) in designs.items():
            if ("bt" if arm == bt else arm.lower()) not in want_sp | {"#words", "interplay"}:
                continue
            r = fit(X, y, train, groups)
            c = (r["pred"] == r["y_test"]).astype(float)
            lo, hi = boot_ci(c)
            nz = np.flatnonzero(r["coef"])
            order = nz[np.argsort(-np.abs(r["coef"][nz]))]
            coefs[arm] = {names[i]: float(r["coef"][i]) for i in order[:TOP_N]}
            rows.append(
                {
                    "arm": arm,
                    "n_features": X.shape[1],
                    "nnz": r["nnz"],
                    "frac_nonzero": r["nnz"] / X.shape[1],
                    "best_C": r["best_C"],
                    "cv_acc": r["cv_acc"],
                    "acc": c.mean(),
                    "ci_lo": lo,
                    "ci_hi": hi,
                }
            )
            print(
                f"  [sparsity] {arm:<14} D={X.shape[1]:>6} nnz={r['nnz']:>5} "
                f"C={r['best_C']:<9g} acc={100 * c.mean():.2f}%"
            )
        pd.DataFrame(rows).sort_values("n_features").to_parquet(
            res / "arm_sparsity.parquet", index=False
        )
        (res / "arm_sparsity_coefs.json").write_text(json.dumps(coefs, indent=2))
        print(f"[text {name}] -> {res}")
    return 0


# --------------------------------------------------------------------------- #
# zero-shot judge
# --------------------------------------------------------------------------- #
def zeroshot_pairs(out_dir=None) -> pd.DataFrame:
    """The heldout pairs with their texts and a seeded slot assignment."""
    z = C.pipeline()["baselines"]["zeroshot"]
    g = C.source_path(C.cell_config(z["pairs_from"])["graph"]["out_dir"])
    items = pd.read_parquet(C.input_path(g / "items.parquet", out_dir))
    pids = sorted(items[items.split == "test"].pair_id.unique())
    d = build_pair_units().set_index("pair_id").loc[pids]
    win_first = np.random.default_rng(int(z["seed"])).random(len(pids)) < 0.5
    return pd.DataFrame(
        {
            "pair_id": pids,
            "op": (d.op_title.fillna("") + "\n" + d.op_body.fillna("")).to_numpy(),
            "winner_text": d.pos_root.to_numpy(),
            "loser_text": d.neg_root.to_numpy(),
            "win_in_A_order1": win_first,
        }
    )


def zeroshot_specs(pairs: pd.DataFrame):
    """Two specs per pair (order 2 mirrors order 1), exactly as the run built them."""
    P = C.prompt("zeroshot")
    specs = []
    for r in pairs.itertuples(index=False):
        for order in (1, 2):
            win_in_A = r.win_in_A_order1 if order == 1 else not r.win_in_A_order1
            a_, b_ = (r.winner_text, r.loser_text) if win_in_A else (r.loser_text, r.winner_text)
            specs.append(
                build_scoring_spec(
                    custom_id=f"{r.pair_id}|o{order}",
                    model=P.MODEL,
                    system=P.SYSTEM,
                    user=P.USER.format(op=r.op, a=a_, b=b_),
                    json_schema=P.SCHEMA,
                    max_output_tokens=P.MAX_OUT,
                    meta={"pair_id": r.pair_id, "order": order, "win_in_A": bool(win_in_A)},
                )
            )
    return specs


def _parse_verdict(text):
    if not text:
        return None, ""
    try:
        o = json.loads(text)
        return o["winner"], o.get("reasoning", "")
    except Exception:  # noqa: BLE001
        m = re.search(r'"winner"\s*:\s*"([AB])"', text or "")
        r = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text or "")
        return (m.group(1) if m else None), (r.group(1) if r else "")


def zeroshot_metrics(v: pd.DataFrame):
    w = v.pivot(index="pair_id", columns="order", values="picked_winner").dropna()
    n = len(w)
    o1, o2 = w[1].astype(bool).to_numpy(), w[2].astype(bool).to_numpy()
    consistent = o1 == o2
    b1 = binomtest(int(o1.sum()), n, 0.5)
    score = np.where(consistent, o1.astype(float), 0.5)
    s = v.pivot(index="pair_id", columns="order", values="slot").loc[w.index]
    aa = int(((s[1] == "A") & (s[2] == "A")).to_numpy()[~consistent].sum())
    bb = int(((s[1] == "B") & (s[2] == "B")).to_numpy()[~consistent].sum())
    bt_slot = binomtest(bb, aa + bb, 0.5) if aa + bb else None
    m = dict(
        n_pairs=n,
        single_order_acc=float(o1.mean()),
        position_named_first=aa,
        position_named_second=bb,
        position_second_share=(bb / (aa + bb)) if aa + bb else float("nan"),
        position_second_ci=(
            [float(bt_slot.proportion_ci().low), float(bt_slot.proportion_ci().high)]
            if bt_slot
            else None
        ),
        position_second_p=float(bt_slot.pvalue) if bt_slot else None,
        single_order_ci=[float(b1.proportion_ci().low), float(b1.proportion_ci().high)],
        single_order_p=float(b1.pvalue),
        order_averaged_acc=float(score.mean()),
        consistency_rate=float(consistent.mean()),
        consistent_subset_acc=float(o1[consistent].mean()) if consistent.any() else float("nan"),
        n_consistent=int(consistent.sum()),
        slot_A_pick_rate=float((v["slot"] == "A").mean()),
    )
    return m, w.index, consistent, score, o1


def zeroshot_compare(score, consistent, o1, idx, out_dir):
    """Paired bootstrap vs the flagship BT arm, plus McNemar on the consistent subset."""
    z = C.pipeline()["baselines"]["zeroshot"]
    shipped = C.cell(z["flagship"]) / "results" / "pair_predictions.parquet"
    pp = pd.read_parquet(C.input_path(shipped, out_dir))
    if not pp.arm.str.startswith("BT-").any():  # a regenerated file may lack the BT arm
        pp = pd.read_parquet(shipped)
    bt = pp[pp.arm.str.startswith("BT-")].set_index("pair_id")
    ours = (bt.loc[idx, "pred"] == bt.loc[idx, "y"]).to_numpy().astype(float)
    diff = ours - score
    rng = np.random.default_rng(0)
    boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(4000)])
    out = {
        "flagship_acc": float(ours.mean()),
        "delta_ours_minus_zeroshot": float(diff.mean()),
        "delta_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
    }
    oc, zc = ours[consistent].astype(int), o1[consistent].astype(int)
    b = int(((oc == 1) & (zc == 0)).sum())
    c = int(((oc == 0) & (zc == 1)).sum())
    out["mcnemar_consistent_subset"] = {
        "n": int(consistent.sum()),
        "b": b,
        "c": c,
        "p": float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0,
    }
    out["by_subset"] = {}
    for lbl, m, zs in [
        ("all", np.ones(len(consistent), bool), score),
        ("judge_self_consistent", consistent, o1.astype(float)),
        ("judge_answered_by_position", ~consistent, np.full(len(consistent), 0.5)),
    ]:
        bb = binomtest(int(ours[m].sum()), int(m.sum()), 0.5)
        out["by_subset"][lbl] = {
            "n": int(m.sum()),
            "zeroshot": float(zs[m].mean()),
            "ours": float(ours[m].mean()),
            "gap": float(ours[m].mean() - zs[m].mean()),
            "ours_ci": [float(bb.proportion_ci().low), float(bb.proportion_ci().high)],
            "ours_p_vs_chance": float(bb.pvalue),
        }
    return out


def cmd_zeroshot(a) -> int:
    J.guard(a.stage, a.yes)
    C.check_writable(a.out_dir, a.in_place)
    z = C.pipeline()["baselines"]["zeroshot"]
    pairs = zeroshot_pairs(a.out_dir)
    if a.stage == "smoke":
        pairs = pairs.head(a.n)
    specs = zeroshot_specs(pairs)
    cache_dir = C.cell_cache(z["cell"])
    if a.stage in ("smoke", "run"):
        from core.judging.batch import run_concurrent
        from core.judging.cache import partition_cached

        _, todo = partition_cached(cache_dir, specs)
        if todo:
            recs, errors = run_concurrent(
                J.client(), todo, cache_dir, max_workers=int(z["workers"])
            )
            ok = [r for r in recs if not r.get("error")]
            print(
                f"[zeroshot] ran {len(recs)}; errors {len(errors)}; "
                f"${actual_usd(ok, batch=False):.2f} sync"
            )
    recs = J.replay(
        specs, cache_dir, allow_missing=a.allow_missing or a.stage == "smoke", label="zeroshot"
    )
    meta = {s.custom_id: s.meta for s in specs}
    rows = []
    for cid, rec in recs.items():
        w, why = _parse_verdict(rec.get("text", ""))
        m = meta[cid]
        rows.append(
            {
                "pair_id": m["pair_id"],
                "order": m["order"],
                "win_in_A": m["win_in_A"],
                "slot": w,
                "picked_winner": None if w is None else (w == "A") == m["win_in_A"],
                "rationale": why,
            }
        )
    v = pd.DataFrame(rows).sort_values(["pair_id", "order"])
    print(f"[zeroshot] {len(v)} verdicts; unparsed {int(v['slot'].isna().sum())}")
    if a.stage == "smoke":
        return 0
    m, idx, consistent, score, o1 = zeroshot_metrics(v)
    m.update(zeroshot_compare(score, consistent, o1, idx, a.out_dir))
    res = C.work_path(C.cell(z["cell"]) / "results", a.out_dir)
    res.mkdir(parents=True, exist_ok=True)
    v.to_parquet(res / "verdicts.parquet", index=False)
    (res / "summary.json").write_text(json.dumps(m, indent=2))
    print(
        f"[zeroshot] single-order {m['single_order_acc']:.4f} | order-averaged (primary) "
        f"{m['order_averaged_acc']:.4f} | consistency {m['consistency_rate']:.4f} | flagship "
        f"{m['flagship_acc']:.4f} -> {res}"
    )
    return 0


# --------------------------------------------------------------------------- #
# embeddings
# --------------------------------------------------------------------------- #
def emb_store(a):
    """The vector store: ``--store``, else the cell's cache/ (written by ``embed``)."""
    e = C.pipeline()["baselines"]["embeddings"]
    from pathlib import Path

    return Path(a.store) if a.store else C.work_path(C.cell(e["cell"]) / "cache", a.out_dir)


def cmd_embeddings(a) -> int:
    e = C.pipeline()["baselines"]["embeddings"]
    J.guard(a.stage, a.yes)
    C.check_writable(a.out_dir, a.in_place)
    store_dir = emb_store(a)
    if a.stage == "embed":
        client = J.client()
        store = EMB.EmbeddingStore(store_dir, model=a.model, dim=a.dim)
        for name in e["splits"]:
            texts = EMB.required_texts(split(name, a.out_dir))
            usage = EMB.embed_missing(store, texts, client, workers=int(e["workers"]))
            print(f"[embed {name}] {usage}")
        return 0

    # fit: refuse cleanly, and without creating anything, when the store is absent
    if not (store_dir / "index.json").exists():
        raise SystemExit(
            f"no embedding vector store at {store_dir}. The embedding_arms cell "
            f"ships without its 1.4 GB cache (raw/cells/README.md), so this arm "
            f"cannot replay offline: pass --store PATH to a copy of the "
            f"cell's embedding cache, or run `embeddings embed --yes` (OpenAI API, ~$2)."
        )
    store = EMB.EmbeddingStore(store_dir, model=a.model, dim=a.dim)
    res = C.work_path(C.cell(e["cell"]) / "results", a.out_dir)
    res.mkdir(parents=True, exist_ok=True)
    for name in e["splits"]:
        mp = split(name, a.out_dir)
        train, groups = mp.train.to_numpy(), mp.op_author.to_numpy()
        if store.missing(EMB.required_texts(mp)):
            raise SystemExit(f"[{name}] the store lacks vectors for this split; run `embed`")
        vec = store.lookup()
        designs, y = {}, None
        for arm, block in (("#words", "words"), ("interplay", "interplay")):
            X, y_ = PT.build_design(
                mp,
                "pos_text",
                "neg_text",
                WF.block_extractor([block])[0],
                op_col="op_text",
                seed=SEED,
            )
            designs[arm], y = (X, [f"{block}:{c}" for c in WF.FEATURE_BLOCKS[block][1]()]), y_
        btd = bt_scores_dir(name, a.out_dir)
        BT = None
        if btd is not None:  # item ids are positional: the BT scores must be the split's cell
            Xbt, fbt = bt_design(mp, btd, seed=SEED)
            BT = f"BT-{len(fbt)}"
            designs[BT] = (Xbt, fbt)
        designs["emb-sim"] = (
            PT.build_design(
                mp, "pos_text", "neg_text", EMB.EmbSimExtractor(vec), op_col="op_text", seed=SEED
            )[0],
            list(EMB.EMB_SIM_NAMES),
        )
        Xe = PT.build_design(
            mp, "pos_text", "neg_text", EMB.EmbExtractor(vec), op_col="op_text", seed=SEED
        )[0]
        designs["emb"] = (Xe, [f"emb:{i}" for i in range(Xe.shape[1])])
        order = ["emb", "emb-sim"] + ([BT] if BT else []) + ["interplay", "#words"]

        fits = {k: fit(designs[k][0], y, train, groups) for k in order}
        yt = fits["#words"]["y_test"]
        corr = {k: (r["pred"] == yt).astype(float) for k, r in fits.items()}
        rows = []
        for k in order:
            lo, hi = boot_ci(corr[k])
            rows.append(
                dict(
                    arm=k,
                    n_features=fits[k]["n_features"],
                    nnz=fits[k]["nnz"],
                    cv_acc=fits[k]["cv_acc"],
                    best_C=fits[k]["best_C"],
                    heldout_acc=float(corr[k].mean()),
                    ci_lo=float(lo),
                    ci_hi=float(hi),
                    delta_vs_interplay=float(corr[k].mean() - corr["interplay"].mean()),
                    mcnemar_vs_interplay=float(
                        PT.mcnemar(yt, fits[k]["pred"], fits["interplay"]["pred"])[2]
                    ),
                    delta_vs_bt=float(corr[k].mean() - corr[BT].mean()) if BT else float("nan"),
                    mcnemar_vs_bt=float(PT.mcnemar(yt, fits[k]["pred"], fits[BT]["pred"])[2])
                    if BT
                    else float("nan"),
                )
            )
            print(f"  [{name}] {k:<10} acc {100 * corr[k].mean():.2f}%  nnz {fits[k]['nnz']}")
        tag = f"n{int(train.sum())}"
        pd.DataFrame(rows).to_parquet(res / f"summary_{tag}.parquet", index=False)
        te_pids = mp.loc[~mp.train, "pair_id"].to_numpy()
        pd.DataFrame(
            [
                dict(pair_id=p, arm=k, pred=int(pr), y=int(yy))
                for k in order
                for p, pr, yy in zip(te_pids, fits[k]["pred"], fits[k]["y_test"])
            ]
        ).to_parquet(res / f"pair_predictions_{tag}.parquet", index=False)

        # per-feature heldout win rates of emb-sim (its joint L1 coefficients are collinear)
        ex = EMB.EmbSimExtractor(vec)
        te = ~mp.train.to_numpy()
        diff = np.array([ex(t, o) for t, o in zip(mp.pos_text[te], mp.op_text[te])]) - np.array(
            [ex(t, o) for t, o in zip(mp.neg_text[te], mp.op_text[te])]
        )
        wr = pd.DataFrame(
            {
                "feature": EMB.EMB_SIM_NAMES,
                "win_rate": [float((c > 0).mean()) for c in diff.T],
                "p": [float(binomtest(int((c > 0).sum()), len(c), 0.5).pvalue) for c in diff.T],
                "coef_joint": fits["emb-sim"]["coef"],
            }
        ).sort_values("win_rate")
        m = len(wr)
        wr["bh_q"] = np.minimum.accumulate(
            (wr.p.sort_values(ascending=False).to_numpy() * m / np.arange(m, 0, -1)).clip(0, 1)
        )[::-1][np.argsort(np.argsort(wr.p.to_numpy()))]
        wr.to_parquet(res / f"emb_sim_winrates_{tag}.parquet", index=False)
        print(f"[embeddings {name}] -> {res}/*_{tag}.parquet")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="what", required=True)
    t = sub.add_parser("text", help="Tan's text arms + sparsity (no LLM)")
    t.add_argument("--cell", action="append", default=None)
    t.add_argument(
        "--rebuild-designs",
        action="store_true",
        help="recompute BOW/POS from text instead of the shipped design_cache",
    )
    t.add_argument("--out-dir", default=None)
    t.add_argument("--in-place", action="store_true")
    z = J.add_stage_args(
        sub.add_parser("zeroshot", help="the zero-shot judge"), stages=("smoke", "run", "collect")
    )
    z.add_argument("--n", type=int, default=8, help="smoke: pairs")
    e = J.add_stage_args(
        sub.add_parser("embeddings", help="text-embedding-3-large arms"), stages=("embed", "fit")
    )
    e.add_argument("--store", default=None, help="the vector store directory")
    e.add_argument("--model", default=EMB.DEFAULT_MODEL)
    e.add_argument("--dim", type=int, default=EMB.DEFAULT_DIM)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.what == "text":
        C.check_writable(a.out_dir, a.in_place)
        return cmd_text(a)
    return {"zeroshot": cmd_zeroshot, "embeddings": cmd_embeddings}[a.what](a)


if __name__ == "__main__":
    raise SystemExit(main())
