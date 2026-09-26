"""Step 6: pool the E subsample rubrics into the M3a/b/c rubrics.

    python scripts/06_pool_rubrics.py --config config.yaml

The M3 baseline pools ``judge.m3_elicitations`` (E, default 27) of the 40-letter rubric
elicitations from step 5, then merges near-duplicate features by agglomerative clustering in
text-embedding-3-large space (average linkage, cosine). Because the clustering threshold is
arbitrary, three cutoffs are run: ``judge.cluster_thresholds`` = 0.55 / 0.60 / 0.65 ->
rubric_stability/pooled_rubric_3{a,b,c}.json (the M3a / M3b / M3c rubrics scored in step 7).

Each cluster collapses to its medoid feature (the one most similar to the rest of its cluster),
keeping that feature's name/description/levels. Needs an API key only for un-cached embeddings.
The M3a/b/c arms are not reported; their scoring token counts calibrate the cost match (06d).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

import _paths  # noqa: F401  (sys.path bootstrap; see scripts/_paths.py)
from core.embeddings import embed_texts

_LETTERS = ["a", "b", "c"]


def _feature_key(f):
    return f"{f['name']}. {f.get('description', '')}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    j = cfg["judge"]
    root = Path(args.config).parent
    stab = root / "results" / "rubric_stability"
    cache_dir = root / ".embed_cache"

    E = int(j.get("m3_elicitations", 27))
    thresholds = list(j.get("cluster_thresholds", [0.55, 0.60, 0.65]))
    rep_files = sorted((stab / "reps").glob("rep_*.json"))[:E]
    if len(rep_files) < E:
        raise SystemExit(
            f"need {E} reps, found {len(rep_files)} — run 05_elicit_rubrics.py reps first"
        )

    feats = [f for rf in rep_files for f in json.loads(rf.read_text())["features"]]
    keys = [_feature_key(f) for f in feats]
    print(f"[pool] pooling {len(feats)} features from E={E} elicitations")

    emb = embed_texts(keys, cache_dir=cache_dir)  # L2-normalized -> cosine = dot
    dist = pdist(emb, metric="cosine")
    Z = linkage(dist, method="average")

    for thr, letter in zip(thresholds, _LETTERS):
        labels = fcluster(Z, t=1.0 - thr, criterion="distance")  # cosine distance cutoff
        merged = []
        for cl in sorted(set(labels)):
            idx = np.where(labels == cl)[0]
            sub = emb[idx]
            medoid = idx[np.argmax((sub @ sub.T).sum(axis=1))]  # most-central feature
            merged.append(feats[medoid])
        out = stab / f"pooled_rubric_3{letter}.json"
        out.write_text(
            json.dumps(
                {
                    "theta_prime": thr,
                    "linkage": "average",
                    "n_elicitations": E,
                    "n_features": len(merged),
                    "features": merged,
                },
                indent=2,
            )
        )
        print(f"[pool] threshold {thr}: {len(feats)} -> {len(merged)} features -> {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
