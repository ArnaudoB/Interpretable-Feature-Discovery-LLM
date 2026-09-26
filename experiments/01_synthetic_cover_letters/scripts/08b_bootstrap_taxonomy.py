"""Step 8b: discovery-count bootstrap for the comparative arm (Ours band of tab:main-bands,
App. app:variance-bands).

    python scripts/08b_bootstrap_taxonomy.py --config config.yaml

Produces the sentence "our method discovered D features, among which our reliability
metrics (kappa_max, rho_min) flagged F unreliable features, giving K reliable features
on average".

Why this is separate from 08_bootstrap.py
-----------------------------------------
`08_bootstrap.py` bootstraps the five RECOVERY metrics. Its comparative arm loads the
taxonomy ONCE before the loop, so the criterion set is fixed and only the kappa/rho
filter varies: it can report `kept`, but "discovered" is a constant there and it can
never produce the discovered/flagged split.

This script re-runs the DISCOVERY step per replicate, which is what the paper describes:

  1. resample the 40 balanced matchings with replacement  (results/gated/taxo_boot_sel.json
     -> `sels[b]`, preserving degree-40 regularity);
  2. use the taxonomy grouping the judge produced for THAT replicate's rationales
     (results/gated/taxo_boot_groupings.json -> `groupings[b]`, indices into `norms[b]`),
     so the number of canonical criteria varies replicate to replicate;
  3. build the design tensor over the resampled pairs, fit, and apply
     kappa <= model.kappa_max and rho > model.rho_threshold.

The grouping calls are LLM calls and are NOT re-issued here: they are shipped, one per
replicate. They were recovered from the realtime response cache -- the batch run
(batch/taxo_boot/) returned only 10 of 50, so do not read batch_output.jsonl for these.
Replicate 28 returned a malformed body and is excluded (see `unparseable`).

Writes results/bootstrap_discovery_counts.json.
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
from core.judging.pricing import actual_usd
from core.model.gated import fit_model

CACHE = None
METRIC_KEYS = ["SF", "SM", "SR", "AF", "LK"]
BAND_KEYS = METRIC_KEYS + ["discovered", "flagged", "kept", "USD"]


def _usage_by_id(root, cache_rel="cache/responses"):
    import glob

    out = {}
    for fp in glob.glob(str(root / cache_rel / "**" / "*.json"), recursive=True):
        try:
            r = json.loads(Path(fp).read_text())
        except (ValueError, OSError):
            continue
        cid = r.get("custom_id")
        u = r.get("usage") or {}
        if cid:
            out[cid] = (r.get("model", ""), u.get("input_tokens", 0), u.get("output_tokens", 0))
    return out


def _usd(entries, pricing=None):
    return actual_usd(
        [{"model": m, "usage": {"input_tokens": i, "output_tokens": o}} for m, i, o in entries],
        pricing,
        batch=True,
    )


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
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    m = cfg["model"]
    root = Path(args.config).parent
    res = root / "results"
    global CACHE
    CACHE = root / ".embed_cache"

    ridge, tol, mi = float(m["ridge"]), float(m["tol"]), int(m["max_iter"])
    kmax, rthr = float(m["kappa_max"]), float(m["rho_threshold"])

    phrases = pd.read_parquet(res / "phrases.parquet")
    pairs = np.load(res / "pairs.npy")
    pair_order = json.loads((res / "pair_order.json").read_text())
    p2i = {str(p): i for i, p in enumerate(pair_order)}
    n = len(json.loads((res / "essay_index.json").read_text()))

    sel = json.loads((res / "gated/taxo_boot_sel.json").read_text())
    doc = json.loads((res / "gated/taxo_boot_groupings.json").read_text())
    NORMS, SELS = sel["norms"], sel["sels"]
    groupings = {int(k): v for k, v in doc["groupings"].items()}
    n_match = int(cfg["graph"]["n_matchings"])
    per_match = len(pairs) // n_match

    eidx = json.loads((res / "essay_index.json").read_text())
    id2l = {i: lid for lid, i in eidx.items()}
    ak = pd.read_parquet(res / "answer_key.parquet").set_index("letter_id")
    dials = relevant_dials(ak)
    G = build_ground_truth(ak)
    theta = float(cfg["metrics"]["theta"])
    tau = float(cfg["metrics"]["tau"])
    usage = _usage_by_id(root)
    pricing = cfg.get("pricing")

    print(
        f"[boot] B={sel['B']} replicates, {len(groupings)} with a usable taxonomy "
        f"(excluded: {doc.get('unparseable')});  kappa<={kmax:g}, rho>{rthr}"
    )

    rows = []
    for b in sorted(groupings):
        norms_b, crits = NORMS[b], groupings[b]
        K = len(crits)
        # phrase -> canonical criterion for THIS replicate's taxonomy
        cof = {}
        for ci, g in enumerate(crits):
            for mem in g["members"]:
                i = mem["index"]
                if 0 <= i < len(norms_b):
                    cof[norms_b[i]] = ci

        X_full = build_X(phrases, cof, K, p2i)  # (T_all, K) under this taxonomy
        idx = np.concatenate(
            [np.arange(p * per_match, (p + 1) * per_match) for p in SELS[b]]
        )  # matchings drawn with replacement
        Xb, pb = X_full[idx], pairs[idx]

        fit = fit_model(Xb, pb, K=K, n=n, ridge=ridge, tol=tol, max_iter=mi)
        kap, rho = kappa_rho(fit, Xb, pb, n, ridge)
        keep_idx = [i for i in range(K) if (kap[i] <= kmax and rho[i] > rthr)]
        kept = len(keep_idx)
        row = {"b": b, "discovered": K, "flagged": K - kept, "kept": kept}

        # recovery metrics under THIS replicate's taxonomy (refit on the kept criteria)
        if kept >= len(dials):
            fr = fit_model(Xb[:, keep_idx], pb, K=kept, n=n, ridge=ridge, tol=tol, max_iter=mi)
            names = [crits[i]["name"] for i in keep_idx]
            ctext = {
                nm: f"{nm}. {crits[i].get('definition', '')}" for i, nm in zip(keep_idx, names)
            }
            F = pd.DataFrame(fr.s, columns=names, index=[id2l[i] for i in range(n)])
            F = F.loc[:, ~F.columns.duplicated()]
            ctext = {c: ctext[c] for c in F.columns}
            M = cosine_matrix(
                [ctext[c] for c in F.columns], [DIAL_TEXT[d] for d in dials], cache_dir=CACHE
            )
            cos = pd.DataFrame(M, index=list(F.columns), columns=dials)
            row.update(metrics.scorecard(F, G.loc[F.index, dials], cos, theta, tau))

        # cost: this replicate's resampled judgments + its own taxonomy call
        ent = [
            usage[f"cmp:{str(pair_order[i]).zfill(6)}"]
            for i in idx
            if f"cmp:{str(pair_order[i]).zfill(6)}" in usage
        ]
        tx = usage.get(f"taxo_boot:b{b:03d}")
        row["USD"] = _usd(ent, pricing) + (_usd([tx], pricing) if tx else 0.0)
        rows.append(row)
        print(
            f"  b={b:03d} discovered={K:2d} kept={kept:2d} "
            f"SR={row.get('SR', float('nan')):.2f} ${row['USD']:.2f}",
            end="\r",
        )
    print()

    df = pd.DataFrame(rows)
    d, f, k = df.discovered, df.flagged, df.kept
    print(f"\n[boot] over B={len(df)} replicates (kappa<={kmax:g}, rho>{rthr}):")
    for name, s in [("discovered", d), ("flagged", f), ("kept", k)]:
        print(
            f"  {name:10s} mean {s.mean():5.2f}  sd {s.std():4.2f}  "
            f"range {s.min():.0f}-{s.max():.0f}"
        )

    band = _band(rows)
    print("\n[boot] bands (mean [2.5, 97.5]) -- taxonomy re-run per replicate:")
    for key in BAND_KEYS:
        if key in band:
            mu, lo, hi = band[key]
            print(f"  {key:11s} {mu:7.3f} [{lo:7.3f}, {hi:7.3f}]")

    out = res / "bootstrap_discovery_counts.json"
    out.write_text(
        json.dumps(
            {
                "B": int(len(df)),
                "kappa_max": kmax,
                "rho_min": rthr,
                "note": "canonical run-variance bootstrap for the comparative arm: the taxonomy "
                "(discovery) step is re-run per replicate, so K varies. Supersedes the "
                "comparative arm of 08_bootstrap.py, which froze the taxonomy.",
                "discovered_mean": float(d.mean()),
                "flagged_mean": float(f.mean()),
                "kept_mean": float(k.mean()),
                "discovered_sd": float(d.std()),
                "flagged_sd": float(f.std()),
                "kept_sd": float(k.std()),
                "bands": {kk: list(vv) for kk, vv in band.items()},
                "per_replicate": rows,
            },
            indent=2,
        )
    )
    print(f"[boot] wrote {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
