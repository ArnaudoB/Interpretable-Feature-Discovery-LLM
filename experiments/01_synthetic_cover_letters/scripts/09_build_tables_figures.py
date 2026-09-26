"""Step 9: compute the recovery metrics for every discovery method (tab:main rows 1-5).

    python scripts/09_build_tables_figures.py --config config.yaml

Computes the five recovery metrics (core.metrics: SF, SM, SR, AF, LK) for every method against
the D=16 role-relevant dials and writes results/tab_results.csv, which step 10 turns into the
paper's tab:main. The figures are built separately, by figures/make_*.py.

  Methods (all vs the same 16 dials):
    comparative ("Ours")  results/scores.parquet          + gated/llm_taxonomy/taxonomy.json text
    M1                     one rep of pointwise/scores_long + rubric_stability/full_dataset_rubric
    M2                     pointwise/scores.parquet (mean)  + full_dataset_rubric
    M3-taxo(1) / (n*)      pointwise_taxo/scores{_long,}.parquet + rubric_stability/taxo_pool/run_000

The clustering fusions M3a/b/c are scored by step 7 only because their token counts calibrate
the cost match (06d); no paper number reports their recovery, so they are not evaluated here.

Also writes results/tracked_dials.json: which dials each method tracks (max |r| > tau), which is
what "Direct LLM--Multi-score still tracks the same 13" rests on.

Semantic matching uses text-embedding-3-large (cached under .embed_cache/, so this runs offline
once seeded); theta=0.406 and tau=0.4 are read from config.metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

import _paths  # noqa: F401  (sys.path bootstrap; see scripts/_paths.py)
from core import metrics
from core.embeddings import cosine_matrix
from dataset.groundtruth import DIAL_TEXT, build_ground_truth, relevant_dials

CACHE = None  # set in main to results-relative .embed_cache
TRACKED: dict = {}  # method -> dials it tracks, filled by _tab_results


def _cos(crit_text: dict, dials) -> pd.DataFrame:
    crits = list(crit_text)
    M = cosine_matrix([crit_text[c] for c in crits], [DIAL_TEXT[d] for d in dials], cache_dir=CACHE)
    return pd.DataFrame(M, index=crits, columns=dials)


def _load_arms(root):
    """Return {method: (scores_df[letter_id-indexed], criterion_text_dict)}."""
    res = root / "results"
    arms = {}

    # comparative ("Ours")
    comp = pd.read_parquet(res / "scores.parquet").set_index("letter_id")
    tax = json.loads((res / "gated/llm_taxonomy/taxonomy.json").read_text())
    ctext = {c["name"]: f"{c['name']}. {c['definition']}" for c in tax["criteria"]}
    arms["comparative"] = (comp, {c: ctext[c] for c in comp.columns if c in ctext})

    # M1 / M2 share the whole-dataset rubric
    full = json.loads((res / "rubric_stability/full_dataset_rubric.json").read_text())
    ftext = {f["name"]: f"{f['name']}. {f['description']}" for f in full["features"]}
    m2 = pd.read_parquet(res / "pointwise/scores.parquet")
    m2.index.name = "letter_id"
    arms["M2"] = (m2, {c: ftext[c] for c in m2.columns if c in ftext})
    long = pd.read_parquet(res / "pointwise/scores_long.parquet")
    m1 = (
        long[long["rep"] == long["rep"].min()]
        .groupby(["letter_id", "dimension"])["score"]
        .mean()
        .unstack()
    )
    arms["M1"] = (m1, {c: ftext[c] for c in m1.columns if c in ftext})

    # M3-taxo — same E=27 elicitations, LLM-taxonomy fusion instead of the clustering cutoff.
    # M3-taxo(1) is rep 0 of the same frame (M1-comparable), exactly as M1 is read off M2's.
    taxo_dir = res / "pointwise_taxo"
    if (taxo_dir / "scores.parquet").exists():
        run0 = json.loads((res / "rubric_stability/taxo_pool/run_000.json").read_text())
        ttext = {f["name"]: f"{f['name']}. {f['description']}" for f in run0["features"]}
        sc = pd.read_parquet(taxo_dir / "scores.parquet")
        sc.index.name = "letter_id"
        arms["M3taxo"] = (sc, {c: ttext[c] for c in sc.columns if c in ttext})
        tl = pd.read_parquet(taxo_dir / "scores_long.parquet")
        s1 = (
            tl[tl["rep"] == tl["rep"].min()]
            .groupby(["letter_id", "dimension"])["score"]
            .mean()
            .unstack()
        )
        arms["M3taxo1"] = (s1, {c: ttext[c] for c in s1.columns if c in ttext})
    return arms


def _tab_results(root, dials, G, theta, tau):
    arms = _load_arms(root)
    rows = []
    labels = {"comparative": "Ours", "M3taxo1": "M3-taxo(1)", "M3taxo": "M3-taxo(n*)"}
    order = [n for n in ["comparative", "M1", "M2", "M3taxo1", "M3taxo"] if n in arms]
    for name in order:
        F, ctext = arms[name]
        common = sorted(set(F.index) & set(G.index))
        Fc = F.loc[common, [c for c in F.columns if c in ctext]]
        Gc = G.loc[common, dials]
        cos = _cos(ctext, dials)
        sc = metrics.scorecard(Fc, Gc, cos, theta, tau)
        sc["method"] = labels.get(name, name)
        rows.append(sc)
        _, _, per_dial = metrics.statistical_recovery(Fc, Gc, tau)
        TRACKED[sc["method"]] = sorted(per_dial.index[per_dial > tau])
    df = pd.DataFrame(rows)[["method", "K", "SF", "SM", "SR", "AF", "LK"]]
    return df, arms


def main() -> int:
    global CACHE
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(args.config).parent
    CACHE = root / ".embed_cache"
    theta = float(cfg["metrics"]["theta"])
    tau = float(cfg["metrics"]["tau"])

    ak = pd.read_parquet(root / "results/answer_key.parquet").set_index("letter_id")
    dials = relevant_dials(ak)  # D = 16 (fluency excluded)
    G = build_ground_truth(ak)
    print(f"[tables] D={len(dials)} relevant dials; theta={theta}, tau={tau}\n")

    tab, arms = _tab_results(root, dials, G, theta, tau)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print("=== tab:results ===")
    print(tab.to_string(index=False))
    (root / "results/tab_results.csv").write_text(tab.to_csv(index=False))
    trk = {
        m: {"n": len(v), "tracked": v, "missed": sorted(set(dials) - set(v))}
        for m, v in TRACKED.items()
    }
    trk["_same_tracked_M1_M2"] = TRACKED["M1"] == TRACKED["M2"]
    (root / "results/tracked_dials.json").write_text(json.dumps(trk, indent=2))
    print(
        f"[tables] tracked dials: "
        + ", ".join(f"{m} {len(v)}" for m, v in TRACKED.items())
        + f"; Direct LLM and its Multi-score variant track the same set: "
        f"{trk['_same_tracked_M1_M2']}"
    )

    print("\n[tables] wrote results/tab_results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
