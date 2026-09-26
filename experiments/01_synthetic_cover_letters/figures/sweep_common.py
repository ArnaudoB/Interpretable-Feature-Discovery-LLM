"""Shared computations for the two acceptance-threshold sweeps of fig:threshold_pair.

Used by make_threshold_pair.py.

The two sweeps use the SAME grid of cosine acceptance thresholds, which is what
lets the pair be read together:
  * surfacing_matrix -- P(dial surfaced in one elicitation) per dial x threshold;
  * pooling_curves   -- features surviving fusion per E x threshold.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform

EXP = Path(__file__).resolve().parent.parent  # the experiment directory
THRESHOLDS = [round(t, 2) for t in np.arange(0.30, 0.81, 0.05)]

DIAL_LABEL = {
    "experience": "Experience",
    "education": "Education level",
    "school_prestige": "School prestige",
    "employer_prestige": "Employer prestige",
    "n_languages": "Language breadth",
    "n_launches": "Production launches",
    "skill_dsa_depth": "Algorithms depth",
    "skill_python": "Python",
    "skill_ai_agents_experience": "Applied AI",
    "skill_security_assessments": "Security assessments",
    "skill_vuln_triage_automation": "Vulnerability automation",
    "skill_agentic_tooling_design": "Agentic tooling",
    "side_project": "Side project",
    "open_source": "Open source",
    "security_cert": "Security certification",
    "ctf": "Capture-the-flag",
}


def _reps():
    files = sorted((EXP / "results/rubric_stability/reps").glob("rep_*.json"))
    return [json.loads(p.read_text()) for p in files]


def surfacing_matrix():
    """(P, labels): P[dial, threshold] = frac. of elicitations surfacing that dial.

    Hungarian-assigns dials to each rep's features ONCE on the full cosine matrix
    (the assignment does not depend on the threshold), then re-thresholds the
    matched cosines. Rows are returned sorted ascending by P at mid-sweep.
    """
    sys.path[:0] = [str(EXP), str(EXP.parents[1])]  # experiment dir + repo root
    from core.embeddings import cosine_matrix
    from dataset.groundtruth import DIAL_TEXT, relevant_dials

    cache = EXP / ".embed_cache"
    ak = pd.read_parquet(EXP / "results/answer_key.parquet").set_index("letter_id")
    dials = relevant_dials(ak)
    dtxt = [DIAL_TEXT[d] for d in dials]

    REPS = _reps()
    BEST = np.zeros((len(REPS), len(dials)))
    for ri, r in enumerate(REPS):
        feats = r["features"]
        if not feats:
            continue
        S = cosine_matrix(
            dtxt, [f"{f['name']}. {f.get('description', '')}" for f in feats], cache_dir=cache
        )
        i_, j_ = linear_sum_assignment(1 - S)
        for i, j in zip(i_, j_):
            BEST[ri, i] = S[i, j]

    P = np.array([[(BEST[:, k] >= t).mean() for t in THRESHOLDS] for k in range(len(dials))])
    order = np.argsort(P[:, THRESHOLDS.index(0.50)])
    return P[order], [DIAL_LABEL[dials[k]] for k in order], len(REPS)


def _feature_space():
    """(emb, DIST, per_rep, dial_cos, dials): everything keyed to one feature index."""
    sys.path[:0] = [str(EXP), str(EXP.parents[1])]  # experiment dir + repo root
    from core.embeddings import cosine_matrix, embed_texts
    from dataset.groundtruth import DIAL_TEXT, relevant_dials

    cache = EXP / ".embed_cache"
    ak = pd.read_parquet(EXP / "results/answer_key.parquet").set_index("letter_id")
    dials = relevant_dials(ak)

    reps = [r["features"] for r in _reps()]
    keys, per_rep = [], []
    for feats in reps:
        idx = []
        for f in feats:
            idx.append(len(keys))
            keys.append(f"{f['name']}. {f.get('description', '')}")
        per_rep.append(np.array(idx, dtype=int))

    emb = embed_texts(keys, cache_dir=cache)
    DIST = np.clip(1.0 - (emb @ emb.T), 0.0, None)
    np.fill_diagonal(DIST, 0.0)
    dial_cos = cosine_matrix([DIAL_TEXT[d] for d in dials], keys, cache_dir=cache)  # (D, N)
    return emb, DIST, per_rep, dial_cos, dials


def dial_recovery_after_fusion(
    E: int = 27, theta: float = 0.406, n_subsets: int = 40, seed: int = 0
):
    """P[dial, threshold] that a dial is RECOVERED, sweeping the FUSION threshold.

    This is the pipeline order, not a matcher sweep:
      1. pool the features of E elicitations;
      2. FUSE them at the x-axis acceptance threshold (average linkage on cosine,
         cut at 1 - t, one medoid per cluster) -- exactly 06_pool_rubrics.py;
      3. Hungarian-match the dials to that fused feature set and accept a match at
         the FIXED calibrated theta (95th pct of rejected-pair cosine).
    Averaged over random E-subsets of the available elicitations.
    """
    emb, DIST, per_rep, dial_cos, dials = _feature_space()
    X = len(per_rep)
    rng = np.random.default_rng(seed)
    hits = {t: np.zeros(len(dials)) for t in THRESHOLDS}

    subsets = [rng.choice(X, size=E, replace=False) for _ in range(n_subsets)]
    for sub in subsets:
        rows = np.concatenate([per_rep[i] for i in sub])
        Z = linkage(squareform(DIST[np.ix_(rows, rows)], checks=False), method="average")
        for t in THRESHOLDS:
            labels = fcluster(Z, t=1.0 - t, criterion="distance")
            medoids = []
            for cl in np.unique(labels):
                idx = rows[labels == cl]
                sub_emb = emb[idx]
                medoids.append(int(idx[np.argmax((sub_emb @ sub_emb.T).sum(axis=1))]))
            S = dial_cos[:, medoids]  # (D, K_fused)
            i_, j_ = linear_sum_assignment(1 - S)
            for i, j in zip(i_, j_):
                if S[i, j] >= theta:
                    hits[t][i] += 1
    P = np.array([hits[t] / len(subsets) for t in THRESHOLDS]).T  # (D, n_thresholds)
    return P, dials


def pooling_curves(n_subsets: int = 40, seed: int = 0):
    """(Es, mean, X): expected features after fusion, per threshold, vs pooling depth E.

    Fusion mirrors 06_pool_rubrics.py (average linkage on cosine distance, cut at
    1 - threshold). The dendrogram is built once per subset and re-cut at every
    threshold, since fcluster only re-cuts an existing linkage.
    """
    sys.path[:0] = [str(EXP), str(EXP.parents[1])]  # experiment dir + repo root
    from core.embeddings import embed_texts

    reps = [r["features"] for r in _reps()]
    X = len(reps)
    keys, per_rep = [], []
    for feats in reps:
        idx = []
        for f in feats:
            idx.append(len(keys))
            keys.append(f"{f['name']}. {f.get('description', '')}")
        per_rep.append(np.array(idx, dtype=int))
    emb = embed_texts(keys, cache_dir=EXP / ".embed_cache")

    DIST = np.clip(1.0 - (emb @ emb.T), 0.0, None)
    np.fill_diagonal(DIST, 0.0)

    def counts(rows):
        if len(rows) < 2:
            return {t: len(rows) for t in THRESHOLDS}
        Z = linkage(squareform(DIST[np.ix_(rows, rows)], checks=False), method="average")
        return {t: int(len(set(fcluster(Z, t=1.0 - t, criterion="distance")))) for t in THRESHOLDS}

    Es = np.arange(1, X + 1)
    rng = np.random.default_rng(seed)
    mean = {t: [] for t in THRESHOLDS}
    for E in Es:
        subsets = (
            [np.arange(X)]
            if E == X
            else [rng.choice(X, size=E, replace=False) for _ in range(n_subsets)]
        )
        acc = {t: [] for t in THRESHOLDS}
        for sub in subsets:
            c = counts(np.concatenate([per_rep[i] for i in sub]))
            for t in THRESHOLDS:
                acc[t].append(c[t])
        for t in THRESHOLDS:
            mean[t].append(float(np.mean(acc[t])))
    return Es, mean, X
