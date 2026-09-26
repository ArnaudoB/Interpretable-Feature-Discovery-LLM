"""Step 10b: held-out accuracy at a matched number of features k — fig:cmv-dimension-curve (right),
fig:cmv-diff-dr, and the matched-k comparisons of Sec. sec:cmv. No LLM calls.

    python scripts/10b_matched_k.py                    # curves + comparisons from the shipped fits
    python scripts/10b_matched_k.py refit --panel "Dr + PW"   # re-fit one panel, compare to shipped

Full-set comparisons conflate feature quality with set size. For k = 1..K, each feature set's
paired L1-LR (step 9's task: 400 train pairs, seed-42 sign deck, GroupKFold by OP; PW sets on
the mean of their two scoring runs) is fitted on up to 50 random k-subsets of its features (all
of them when there are at most 50), and the subset-averaged held-out accuracies are compared on
the same 800 pairs (``core.cmv.matched_k``). Intervals: joint bootstrap over the held-out pairs
and the drawn subsets, 4,000 replicates, one pair resample shared by every set and every k.

**Inputs.** The per-subset fits (``raw/cells/matched_k/fits/<SET>_k<k>.npz``: 0/1 held-out
correctness per subset, and the subsets themselves) are shipped alongside the LLM response
caches and read by default; ``refit --panel <SET>`` recomputes them and reports the agreement
with the shipped ones. Everything downstream of the fits -- curves, differences, summaries,
figures -- is deterministic and is recomputed here.

Writes results/matched_k/accuracy_curves.parquet and, per comparison,
results/matched_k/<REF>_vs_<OTHER>/{matched_k_curve.parquet, summary.json}.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
import zlib

import numpy as np
import pandas as pd

import _paths  # noqa: F401
import _config as C
from core.cmv import matched_k as MK
from core.cmv import pairtask as PT
from core.cmv.designs import fit_select, signed_diff_design
from core.cmv.paired import average_runs, boot_indices

warnings.filterwarnings("ignore")  # sklearn's penalty= deprecation flood

FITS = C.CELLS / "matched_k" / "fits"
OUT = C.RESULTS / "matched_k"
M_SUBSETS, N_BOOT, SEED = 50, 4000, 0
K_READ = 16  # the matched reading point quoted in the text

#: The comparisons the paper reports, as (ref, other): the difference is ref - other.
#: The first two back "D_r beats D at every k"; the third "S is 2.9 points below D + PW";
#: the last two complete the four panels of fig:cmv-diff-grid.
COMPARISONS = [
    ("Dr + PW", "D + PW"),
    ("Dr + BT", "D + BT"),
    ("D + PW", "S + PW"),
    ("Dr + PW", "S + PW"),
    ("Dr + BT", "S + PW"),
]


def tag(arm: str) -> str:
    return arm.replace(" + ", "").upper()  # "Dr + PW" -> "DRPW", the fit-file prefix


def load_fits(arm: str) -> tuple[dict, dict]:
    corr, exact = {}, {}
    for f in sorted(FITS.glob(f"{tag(arm)}_k*.npz")):
        z = np.load(f)
        k = int(f.stem.split("_k")[1])
        corr[k], exact[k] = z["correct"], bool(z["exact"])
    assert corr, f"no fits for {arm} under {FITS}"
    return corr, exact


def accuracy_curves() -> pd.DataFrame:
    """Every set's subset-averaged accuracy vs k, one shared pair resample."""
    idx = boot_indices(800, N_BOOT, SEED)
    out = []
    for arm in C.SCORE_ARMS:
        corr, exact = load_fits(arm)
        c = MK.marginal_curve(corr, exact, idx, seed=SEED).assign(arm=arm)
        out.append(c)
        print(
            f"  {arm:<8} k=1..{int(c.k.max()):>2}  k=16 {100 * c.loc[c.k == 16, 'acc'].iloc[0]:5.2f}  "
            f"k={int(c.k.max())} {100 * c.acc.iloc[-1]:5.2f} [{100 * c.lo.iloc[-1]:.1f}, {100 * c.hi.iloc[-1]:.1f}]"
        )
    df = pd.concat(out, ignore_index=True)
    df.to_parquet(OUT / "accuracy_curves.parquet", index=False)
    return df


def compare(ref: str, other: str) -> dict:
    corr, exact = {}, {}
    ks = None
    for arm in (ref, other):
        c, e = load_fits(arm)
        ks = set(c) if ks is None else ks & set(c)
        corr |= {(arm, k): v for k, v in c.items()}
        exact |= {(arm, k): v for k, v in e.items()}
    kmax = max(ks)
    corr = {key: v for key, v in corr.items() if key[1] <= kmax}
    curve, D = MK.matched_k_bootstrap(
        corr, exact, (ref, other), boot_indices(800, N_BOOT, SEED), seed=SEED
    )
    ks = curve.k.tolist()
    summary = {
        "difference": f"{ref} minus {other}",
        "k": [ks[0], ks[-1]],
        "n_boot": N_BOOT,
        "q_simul": curve.attrs["q_simul"],
        "primary_mean_over_k": MK.summarize(curve, D),
        f"k{K_READ}": MK.summarize(curve, D, cols=[ks.index(K_READ)]) if K_READ in ks else None,
        "k_simul_excludes_0": curve.loc[(curve.hi_simul < 0) | (curve.lo_simul > 0), "k"].tolist(),
        "k_pointwise_excludes_0": curve.loc[(curve.hi < 0) | (curve.lo > 0), "k"].tolist(),
    }
    d = OUT / f"{tag(ref)}_vs_{tag(other)}"
    d.mkdir(parents=True, exist_ok=True)
    curve.to_parquet(d / "matched_k_curve.parquet", index=False)
    (d / "summary.json").write_text(json.dumps(summary, indent=2))
    p = summary["primary_mean_over_k"]
    print(
        f"  {ref} - {other}: mean over k=1..{kmax} {100 * p['estimate']:+.2f} "
        f"[{100 * p['lo']:+.2f}, {100 * p['hi']:+.2f}]; delta > 0 at every k: "
        f"{bool((curve.delta > 0).all())}"
    )
    return summary


def refit(arm: str) -> int:
    """Re-fit one set's subsets and report agreement with the shipped fits. Never overwrites."""
    task = C.cfg()["task"]
    mp = C.matched_pairs()
    train, groups = mp.train.to_numpy(), mp[task["group_by"]].to_numpy()
    signs = PT.assign_signs(mp, seed=int(task["seed"]))
    y = (signs == 1).astype(int)
    run_a, run_b = C.SCORE_ARMS[arm]
    S = C.panel(run_a)
    if run_b is not None:
        S = average_runs(S, C.panel(run_b))
    feats = list(S.columns)
    np.random.seed(int(task["liblinear_global_seed"]))
    rng = np.random.default_rng([SEED, zlib.crc32(arm.encode())])
    shipped, _ = load_fits(arm)
    t0, same, total, dpairs = time.time(), 0, 0, []
    for k in range(1, len(feats) + 1):
        subs, _ = MK.draw_subsets(feats, k, M_SUBSETS, rng)
        ref_subs = json.loads(str(np.load(FITS / f"{tag(arm)}_k{k:02d}.npz")["subsets"]))
        assert subs == ref_subs, f"{arm} k={k}: subset draw differs from the shipped fits"
        for i, sub in enumerate(subs):
            r = fit_select(signed_diff_design(mp, S, sub, signs), y, train, groups, cs=C.c_grid())
            got = (r["pred"] == r["y_test"]).astype(np.int8)
            same += int((got == shipped[k][i]).all())
            total += 1
            dpairs.append(int(got.sum()) - int(shipped[k][i].sum()))
        print(
            f"  k={k:>2} {len(subs):>2} subsets: {same}/{total} identical so far "
            f"[{time.time() - t0:.0f}s]",
            flush=True,
        )
    dp = np.abs(dpairs)
    print(
        f"[refit] {arm}: {same}/{total} subset fits identical to the shipped ones; "
        f"|correct-count difference| max {dp.max()}, mean {dp.mean():.3f}"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", nargs="?", default="curves", choices=["curves", "refit"])
    ap.add_argument("--panel", choices=list(C.SCORE_ARMS))
    a = ap.parse_args()
    if a.stage == "refit":
        assert a.panel, "refit needs --panel"
        return refit(a.panel)
    OUT.mkdir(parents=True, exist_ok=True)
    print("[matched-k] accuracy curves")
    accuracy_curves()
    print("[matched-k] comparisons")
    for ref, other in COMPARISONS:
        compare(ref, other)
    print("[matched-k] wrote results/matched_k/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
