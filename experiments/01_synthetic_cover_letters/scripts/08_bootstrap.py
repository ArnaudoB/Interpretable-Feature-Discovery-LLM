"""Step 8: run-variance bootstrap of the Direct LLM arm (code M1) -- tab:main-bands, tab:paired (a).

    python scripts/08_bootstrap.py --config config.yaml

Resamples the data each method consumes, re-executes its discovery step, and recomputes the five
metrics; theta and tau are held fixed (B = config.bootstrap.B, default 50):

  * M1   -- resample the 200 letters with replacement, re-elicit the whole-dataset rubric and
            re-score on that resample (both shipped: gated/m1_boot_rubrics.json + the
            batch/m1_boot_score outputs), evaluate on the resampled multiset.

Our own method's run variance comes from 08b (matchings, taxonomy re-run) and 08d (letters).

Prints/saves tab_runvar.json (mean [2.5, 97.5] percentile band per metric). Embeddings are cached
under .embed_cache/, so after one seeding run this is offline.
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
from dataset.groundtruth import DIAL_TEXT, build_ground_truth, relevant_dials

CACHE = None
METRIC_KEYS = ["SF", "SM", "SR", "AF", "LK"]
# Banded alongside the metrics: K (already returned by metrics.scorecard) and USD, the
# per-replicate cost on a BATCH basis (see 06d_cost_match.py for why batch throughout).
BAND_KEYS = METRIC_KEYS + ["K", "USD"]


def _cos(crit_text, dials):
    crits = list(crit_text)
    M = cosine_matrix([crit_text[c] for c in crits], [DIAL_TEXT[d] for d in dials], cache_dir=CACHE)
    return pd.DataFrame(M, index=crits, columns=dials)


def _usage_by_id(root, cache_rel="cache/responses"):
    """{custom_id: (model, in_tok, out_tok)} from the shipped response cache."""
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
    """Batch-basis USD for an iterable of (model, in_tok, out_tok)."""
    from core.judging.pricing import actual_usd

    return actual_usd(
        [{"model": m, "usage": {"input_tokens": i, "output_tokens": o}} for m, i, o in entries],
        pricing,
        batch=True,
    )


def _band(rows):
    """rows: list of dicts of metric->value -> {metric: (mean, lo, hi)}."""
    out = {}
    for k in BAND_KEYS:
        vals = [r[k] for r in rows if r.get(k) is not None and not np.isnan(r.get(k, np.nan))]
        if not vals:
            continue
        v = np.array(vals, float)
        out[k] = (float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return out


def _parse_m1_scores(root):
    """{b: {letter_id: {dimension: score}}} from batch/m1_boot_score/batch_output.jsonl."""

    def _find(o):
        if isinstance(o, dict):
            if "scores" in o and isinstance(o["scores"], list):
                return o
            for v in o.values():
                r = _find(v)
                if r:
                    return r
        elif isinstance(o, list):
            for v in o:
                r = _find(v)
                if r:
                    return r
        elif isinstance(o, str) and '"scores"' in o:
            try:
                return json.loads(o)
            except json.JSONDecodeError:
                return None
        return None

    out: dict[int, dict] = {}
    fp = root / "batch/m1_boot_score/batch_output.jsonl"
    for line in fp.open():
        d = json.loads(line)
        cid = d.get("custom_id", "")  # m1score:b000:L0001
        parts = cid.split(":")
        if len(parts) != 3:
            continue
        b = int(parts[1][1:])
        lid = parts[2]
        payload = _find(d.get("response", {}).get("body", {}))
        if not payload:
            continue
        out.setdefault(b, {})[lid] = {s["dimension"]: s["score"] for s in payload["scores"]}
    return out


def _bootstrap_m1(root, dials, GT, theta, tau, B, pricing=None):
    res = root / "results"
    resamp = json.loads((res / "gated/m1_boot_resamples.json").read_text())
    rubrics = json.loads((res / "gated/m1_boot_rubrics.json").read_text())
    letter_ids = resamp["letter_ids"]
    resamples = resamp["resamples"]
    scores = _parse_m1_scores(root)
    usage = _usage_by_id(root)

    rows = []
    for b in range(min(B, len(resamples))):
        feats = rubrics[str(b)]
        ctext = {f["name"]: f"{f['name']}. {f.get('description', '')}" for f in feats}
        dims = list(ctext)
        sc_b = scores.get(b, {})
        table = pd.DataFrame({lid: sc_b.get(lid, {}) for lid in sc_b}).T.reindex(columns=dims)
        drawn = [letter_ids[i] for i in resamples[b] if letter_ids[i] in table.index]
        if len(drawn) < 20:
            continue
        F = table.loc[drawn].reset_index(drop=True).astype(float)
        Gc = GT.loc[drawn, dials].reset_index(drop=True)
        row = metrics.scorecard(F, Gc, _cos(ctext, dials), theta, tau)
        ent = [
            v
            for k, v in usage.items()
            if k == f"m1boot_elic:b{b:03d}" or k.startswith(f"m1score:b{b:03d}:")
        ]
        row["USD"] = _usd(ent, pricing) if ent else float("nan")
        rows.append(row)
        print(
            f"  [m1]  b={b + 1}/{B} K={F.shape[1]} SF={row['SF']:.2f} ${row['USD']:.2f}", end="\r"
        )
    print()
    return _band(rows), rows


def _fmt(band):
    return "  ".join(
        f"{k} {band[k][0]:.2f}[{band[k][1]:.2f},{band[k][2]:.2f}]" for k in BAND_KEYS if k in band
    )


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
    B = int(cfg["bootstrap"]["B"])

    ak = pd.read_parquet(root / "results/answer_key.parquet").set_index("letter_id")
    dials = relevant_dials(ak)
    G = build_ground_truth(ak)

    print(f"[bootstrap] B={B}, D={len(dials)} dials, theta={theta}, tau={tau}")
    print("[bootstrap] M1 ...")
    m1, m1_rows = _bootstrap_m1(root, dials, G, theta, tau, B, cfg.get("pricing"))
    print("\n=== run variance, Direct LLM (M1) ===")
    print(f"M1    {_fmt(m1)}")
    out = {"B": B, "M1": m1, "M1_per_replicate": m1_rows}
    (root / "results/tab_runvar.json").write_text(json.dumps(out, indent=2))
    print("\n[bootstrap] wrote results/tab_runvar.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
