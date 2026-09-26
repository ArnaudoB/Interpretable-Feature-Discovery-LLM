"""Step 8d: letter-resampled bootstrap for the comparative arm — comparable to method 1a.

    python scripts/08d_bootstrap_ours_letters.py --config config.yaml

``08b`` resamples MATCHINGS, so the 200 letters are identical in every replicate (each matching
is perfect: every letter appears exactly once). Its band therefore covers estimation noise given
this corpus, but NOT corpus-sampling noise. Method 1a's bootstrap resamples LETTERS, so its band
contains a variance component Ours' does not — the asymmetry flatters Ours.

This script removes that asymmetry by resampling the same unit, reusing the SAME 50 letter
resamples 1a was bootstrapped on (results/gated/m1_boot_resamples.json), which makes the two
arms **paired**: replicate b is the same draw in both, so per-replicate differences are meaningful.

One deliberate handicap against Ours: only cached judgments are used, so a pair survives only
if BOTH endpoints were drawn — ~1600 of 4000 pairs, an effective degree of ~25 against the
designed 40. The taxonomy is re-run per letter resample (07c supplies one grouping per resample),
which is the default here; ``--frozen-taxonomy`` holds it at the canonical fit instead and
writes results/bootstrap_ours_letters_frozen.json.

The fit uses the induced subgraph on the distinct drawn letters (each surviving comparison
counted once — duplicating identical rows would inflate the Fisher information and distort the
kappa/rho filter). The METRICS are then evaluated on the drawn multiset, duplicates included,
exactly as 1a does, so the two are computed the same way.

Writes results/bootstrap_ours_letters.json. Feeds tab:paired (a), fig:paired (a) and
App. app:paired-design.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import _paths  # noqa: F401  (sys.path bootstrap; see scripts/_paths.py)
from core import metrics
from core.embeddings import cosine_matrix
from core.fit import build_X, kappa_rho
from dataset.groundtruth import DIAL_TEXT, build_ground_truth, relevant_dials
from core.model.gated import fit_model

BAND_KEYS = ["SF", "SM", "SR", "AF", "LK", "K", "n_pairs", "n_letters", "degree"]


def _band(rows):
    out = {}
    for k in BAND_KEYS:
        v = np.array(
            [r[k] for r in rows if r.get(k) is not None and not np.isnan(float(r.get(k, np.nan)))],
            float,
        )
        if len(v):
            out[k] = (float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument(
        "--frozen-taxonomy",
        action="store_true",
        help="opt out of per-replicate discovery and reuse the single canonical "
        "taxonomy. NOT the default: freezing understates this arm and breaks "
        "symmetry with 1a, whose bootstrap re-elicits its rubric every replicate.",
    )
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    m = cfg["model"]
    root = Path(args.config).parent
    res = root / "results"
    cache = root / ".embed_cache"
    theta = float(cfg["metrics"]["theta"])
    tau = float(cfg["metrics"]["tau"])
    ridge, tol, mi = float(m["ridge"]), float(m["tol"]), int(m["max_iter"])
    kmax, rthr = float(m["kappa_max"]), float(m["rho_threshold"])

    phrases = pd.read_parquet(res / "phrases.parquet")
    pairs = np.load(res / "pairs.npy")
    pair_order = json.loads((res / "pair_order.json").read_text())
    p2i = {str(p): i for i, p in enumerate(pair_order)}
    eidx = json.loads((res / "essay_index.json").read_text())
    n = len(eidx)
    id2l = {i: lid for lid, i in eidx.items()}

    mapping = json.loads((res / "gated/llm_taxonomy/mapping.json").read_text())
    crit_names = sorted(set(mapping.values()))
    K = len(crit_names)
    cof = {norm: crit_names.index(c) for norm, c in mapping.items()}
    tax = json.loads((res / "gated/llm_taxonomy/taxonomy.json").read_text())
    ctext_all = {c["name"]: f"{c['name']}. {c['definition']}" for c in tax["criteria"]}
    X = build_X(phrases, cof, K, p2i)

    ak = pd.read_parquet(res / "answer_key.parquet").set_index("letter_id")
    dials = relevant_dials(ak)
    G = build_ground_truth(ak)

    lb = None
    if not args.frozen_taxonomy:
        f = res / "gated/letterboot_groupings.json"
        if not f.exists():
            raise SystemExit("run 07c_taxonomy_letter_boot.py collect first")
        doc = json.loads(f.read_text())
        lb = (
            {int(k): v for k, v in doc["groupings"].items()},
            {int(k): v for k, v in doc["norms"].items()},
        )
        print(f"[ours-letters] per-replicate taxonomy: {len(lb[0])} groupings from 07c")

    resamp = json.loads((res / "gated/m1_boot_resamples.json").read_text())
    letter_ids, resamples = resamp["letter_ids"], resamp["resamples"]
    print(
        f"[ours-letters] {len(resamples)} letter resamples (shared with 1a), "
        f"frozen taxonomy K={K}, kappa<={kmax:g}, rho>{rthr}"
    )

    rows = []
    for b, draw in enumerate(resamples):
        drawn_lids = [letter_ids[i] for i in draw]  # multiset, duplicates kept
        mult = np.bincount([eidx[l] for l in drawn_lids], minlength=n)
        inset = mult > 0
        keep = inset[pairs[:, 0]] & inset[pairs[:, 1]]  # both endpoints drawn
        idx = np.flatnonzero(keep)

        # Restrict the model to letters that actually appear. Keeping the absent ~73 letters as
        # free parameters held only by the ridge leaves the Hessian near-singular, which blows up
        # every per-criterion condition number and empties the kappa/rho filter.
        if lb is not None:
            if b not in lb[0]:
                print(f"  b={b:03d} no taxonomy for this replicate — skipped")
                continue
            crits_b, norms_b = lb[0][b], lb[1][b]
            K_b = len(crits_b)
            cof_b = {
                norms_b[mem["index"]]: ci
                for ci, g_ in enumerate(crits_b)
                for mem in g_.get("members", [])
                if 0 <= mem["index"] < len(norms_b)
            }
            X_b = build_X(phrases, cof_b, K_b, p2i)
            names_all = [c["name"] for c in crits_b]
            ctext_b = {c["name"]: f"{c['name']}. {c.get('definition', '')}" for c in crits_b}
        else:
            X_b, K_b, names_all, ctext_b = X, K, crit_names, ctext_all

        sub = np.flatnonzero(inset)
        remap = np.full(n, -1, dtype=int)
        remap[sub] = np.arange(len(sub))
        n_sub = int(len(sub))
        Xb, pb = X_b[idx], remap[pairs[idx]]

        fit = fit_model(Xb, pb, K=K_b, n=n_sub, ridge=ridge, tol=tol, max_iter=mi)
        kap, rho = kappa_rho(fit, Xb, pb, n_sub, ridge)
        keep_idx = [i for i in range(K_b) if (kap[i] <= kmax and rho[i] > rthr)]
        # Do NOT skip replicates with K < D. metrics.hungarian_match takes a rectangular
        # (D, K) cost matrix and simply leaves D-K dials unassigned, which semantic_recovery
        # scores as misses -- the correct penalty. Dropping these would condition on success
        # and bias every band upward, since they are exactly the weakest replicates.
        if len(keep_idx) < 2:
            print(f"  b={b:03d} only {len(keep_idx)} criteria survived — unusable, skipped")
            continue
        fr = fit_model(
            Xb[:, keep_idx], pb, K=len(keep_idx), n=n_sub, ridge=ridge, tol=tol, max_iter=mi
        )
        names = [names_all[i] for i in keep_idx]
        S = pd.DataFrame(fr.s, columns=names, index=[id2l[g] for g in sub])
        ctext = {c: ctext_b[c] for c in names if c in ctext_b}
        S = S[list(ctext)]

        # evaluate on the drawn MULTISET, exactly as 1a does
        F = S.loc[drawn_lids].reset_index(drop=True)
        Gc = G.loc[drawn_lids, dials].reset_index(drop=True)
        M = cosine_matrix(
            [ctext[c] for c in S.columns], [DIAL_TEXT[d] for d in dials], cache_dir=cache
        )
        cos = pd.DataFrame(M, index=list(S.columns), columns=dials)
        row = metrics.scorecard(F, Gc, cos, theta, tau)
        row.update(
            {
                "b": b,
                "n_pairs": int(len(idx)),
                "n_letters": int(inset.sum()),
                "degree": 2.0 * len(idx) / max(int(inset.sum()), 1),
            }
        )
        rows.append(row)
        print(
            f"  b={b:03d} pairs={len(idx):4d} letters={int(inset.sum())} "
            f"deg={row['degree']:.1f} K={row['K']:2d} SR={row['SR']:.2f}",
            end="\r",
        )
    print()

    band = _band(rows)
    print(f"\n[ours-letters] B={len(rows)} usable replicates; bands (mean [2.5, 97.5]):")
    for k in BAND_KEYS:
        if k in band:
            mu, lo, hi = band[k]
            print(f"  {k:9s} {mu:8.3f} [{lo:8.3f}, {hi:8.3f}]")

    # canonical = per-replicate discovery; the frozen variant is kept only as a contrast
    out = res / (
        "bootstrap_ours_letters_frozen.json"
        if args.frozen_taxonomy
        else "bootstrap_ours_letters.json"
    )
    out.write_text(
        json.dumps(
            {
                "B": len(rows),
                "frozen_taxonomy": bool(args.frozen_taxonomy),
                "K_taxonomy": (K if args.frozen_taxonomy else None),
                "resampling_unit": "letter (shared with 1a -> paired)",
                "resample_source": "results/gated/m1_boot_resamples.json",
                "caveat": "cached judgments only: a pair survives iff both endpoints were drawn, so the "
                "effective degree is ~25 vs 40. Both this and the frozen taxonomy bias AGAINST "
                "the comparative arm.",
                "bands": {k: list(v) for k, v in band.items()},
                "per_replicate": rows,
            },
            indent=2,
        )
    )
    print(f"[ours-letters] wrote {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
