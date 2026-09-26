"""Pairwise design construction for the CMV predictive task (Tan text arms + the fitted BT panel).

Every entry point that needs a design builds it here, so all of them build identical designs
and their heldout numbers stay comparable.

Every arm's design is the signed pair difference ``x = sign * (f_pos - f_neg)`` with the sign
dealt by :func:`core.cmv.pairtask.assign_signs` at the shared seed 42, which is what
makes paired McNemar tests valid *across* arms.

Memory: the full-train BOW design densifies to ~4 GB, so sparse arms stay sparse end to end.
Nothing here calls ``.toarray()``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, normalize

from core.cmv import features as WF
from core.cmv import pairtask as PT
from core.cmv.data import CONDITIONS, build_pair_units
from core.cmv.items import build_matched_items

SEED = 42
CS = np.logspace(-3, 3, 13)

#: Display order used by both entry points. ``BT-K`` is inserted at the front when present.
ARM_ORDER = ["#words", "interplay", "interplay+POS", "BOW", "POS", "all-Tan"]


# --------------------------------------------------------------------------- #
# split
# --------------------------------------------------------------------------- #
def matched_pairs_from_split(cfg, repo):
    """Build the flagship's matched pairs (root text + train/test split) with no graph/LLM.

    ``repo`` is the repository root; paths in the config are resolved against it (no path is
    hardcoded in this module).
    """
    repo = Path(repo)
    scfg = cfg["source"]
    cond = scfg.get("challenge_field", "full_path")
    pos_col, neg_col = CONDITIONS[cond]
    pu = build_pair_units()
    include = None
    if scfg.get("discovery_items"):
        dpath = repo / scfg["discovery_items"]
        if dpath.exists():
            include = pd.read_parquet(dpath)["pair_id"].unique().tolist()
    items = build_matched_items(
        pu,
        n_pairs=int(scfg["n_pairs"]),
        include_pair_ids=include,
        test_frac=float(scfg.get("test_frac", 0.3)),
        min_words=int(scfg["min_words"]),
        max_words=int(scfg["max_words"]),
        seed=int(scfg["seed"]),
        condition=cond,
        test_from_heldout=bool(scfg.get("test_from_heldout", False)),
        n_test=(int(scfg["n_test"]) if scfg.get("n_test") is not None else None),
    )
    pud = pu.set_index("pair_id")
    rows = []
    for pid, grp in items.groupby("pair_id"):
        w = grp[grp.delta == 1].iloc[0]
        l = grp[grp.delta == 0].iloc[0]
        r = pud.loc[pid]
        rows.append(
            dict(
                pair_id=pid,
                split=w.split,
                op_author=w.op_author,
                winner_id=w.item_id,
                loser_id=l.item_id,
                op_text=(r.op_title or "") + "\n" + (r.op_body or ""),
                pos_text=r[pos_col],
                neg_text=r[neg_col],
            )
        )
    mp = pd.DataFrame(rows)
    mp["train"] = mp.split == "train"
    return mp, cond


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def fit_select(X, y, train, groups, cs=CS, n_jobs=None, random_state=None):
    """L1-logreg, grouped 5-fold CV on train, single heldout pass. Returns a result dict.

    Sparse X stays sparse (the mean imputer would densify, so it is only used for the dense text
    designs, which can carry NaN).

    ``random_state`` is passed to liblinear, which otherwise shuffles coordinates from an
    unseeded RNG, so the fit depends on the RNG state (about one held-out pair, and possibly the
    selected C). Set it for a run independent of numpy's global RNG; steps 8-10b instead seed
    that global RNG once (``task.liblinear_global_seed``) and fit in a fixed order.
    """
    steps = [] if sparse.issparse(X) else [("i", SimpleImputer(strategy="mean"))]
    steps += [
        ("s", StandardScaler(with_mean=False)),
        (
            "c",
            LogisticRegression(
                penalty="l1",
                solver="liblinear",
                max_iter=5000,
                fit_intercept=False,
                random_state=random_state,
            ),
        ),
    ]
    # n_jobs=1 for sparse/high-dim: parallel folds copy the design per worker (OOM risk).
    if n_jobs is None:
        n_jobs = 1 if (sparse.issparse(X) or X.shape[1] > 500) else -1
    gs = GridSearchCV(
        Pipeline(steps), {"c__C": list(cs)}, cv=GroupKFold(5), scoring="accuracy", n_jobs=n_jobs
    )
    gs.fit(X[train], y[train], groups=groups[train])
    coef = gs.best_estimator_.named_steps["c"].coef_.ravel()
    return {
        "pred": gs.predict(X[~train]),
        "y_test": y[~train],
        "cv_acc": float(gs.best_score_),
        "best_C": float(gs.best_params_["c__C"]),
        "nnz": int(np.count_nonzero(coef)),
        "n_features": int(X.shape[1]),
        # On the standardized scale, so magnitudes are comparable across columns. Only
        # meaningful for low-dimensional arms; callers ignore it for BOW/all-Tan.
        "coef": coef,
    }


def run_lr(X, y, train, groups):
    """Back-compat shim: ``(pred, y_heldout)`` only. See :func:`fit_select`."""
    r = fit_select(X, y, train, groups)
    return r["pred"], r["y_test"]


# --------------------------------------------------------------------------- #
# text vectorizers
# --------------------------------------------------------------------------- #
def unit_vectors(mp, kind):
    """item_id -> L2-normalized TF vector (BOW words or spaCy POS tags), vocab fit on TRAIN units.

    Returns ``(vectors, vocab)``; ``vocab`` is the ordered token list backing the columns.
    """
    item_text, item_split = {}, {}
    for _, r in mp.iterrows():
        item_text[r.winner_id] = r.pos_text
        item_text[r.loser_id] = r.neg_text
        item_split[r.winner_id] = r.split
        item_split[r.loser_id] = r.split
    ids = list(item_text)
    docs = [item_text[i] for i in ids]
    if kind == "pos":
        import spacy

        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
        docs = [" ".join(t.tag_ for t in d) for d in nlp.pipe(docs, batch_size=64)]
    train_docs = [docs[k] for k, i in enumerate(ids) if item_split[i] == "train"]
    cv = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b")
    counts = np.asarray(cv.fit_transform(train_docs).sum(0)).ravel()
    vocab = sorted(cv.vocabulary_, key=lambda t: cv.vocabulary_[t])
    keep = [vocab[j] for j in range(len(vocab)) if counts[j] > 5] if kind == "bow" else vocab
    cv2 = CountVectorizer(vocabulary=keep, lowercase=True, token_pattern=r"(?u)\b\w+\b")
    M = normalize(cv2.transform(docs).astype(float))
    return {i: M[k] for k, i in enumerate(ids)}, keep


def design_from_vecs(mp, vecs, seed=SEED):
    """Sparse (fw - fl) * sign design. Stays sparse: densifying is a real OOM at full-train BOW
    (4k pairs x ~30k vocab x 8B x 2 ~= 4GB)."""
    fw = sparse.vstack([vecs[i] for i in mp.winner_id], format="csr")
    fl = sparse.vstack([vecs[i] for i in mp.loser_id], format="csr")
    s = PT.assign_signs(mp, seed=seed)  # shared with PT.build_design -> McNemar-valid
    X = sparse.csr_matrix((fw - fl).multiply(s[:, None]))
    return X, (s == 1).astype(int)


def resid_betas(X, w, train):
    """Per-column OLS slope of ``X`` on ``w``, fit on TRAIN, no intercept.

    No intercept on purpose: the design is antisymmetric under ``x -> -x`` / ``y -> 1-y``
    and the classifier sets ``fit_intercept=False``, so adding one would break that
    symmetry. Residualizing is then ``X - np.outer(w, resid_betas(X, w, train))``.

    Kept in the library so every caller that partials out length uses the same arithmetic.
    """
    wt = w[train]
    return (np.asarray(X[train], dtype=float).T @ wt) / float(wt @ wt)


def signed_diff_design(mp, S, feats, signs):
    """``(winner - loser) * sign`` design from an arbitrary item x feature score frame.

    The arithmetic of :func:`bt_design`, separated out so an arm can supply its own scores --
    a pointwise panel, a two-run mean, a subset of criteria -- instead of reading one cell's
    ``scores_{anchor,test}.parquet``. ``signs`` must come from
    :func:`core.cmv.pairtask.assign_signs` at the shared seed, or paired McNemar tests across
    arms stop being valid.
    """
    fw = S.loc[mp.winner_id, feats].to_numpy(float)
    fl = S.loc[mp.loser_id, feats].to_numpy(float)
    return (fw - fl) * signs[:, None]


def bt_design(mp, results_dir, seed=SEED):
    """Fitted BT panel design from ``results/scores_{anchor,test}.parquet``. Dense, K ~ 16."""
    results_dir = Path(results_dir)
    sA = pd.read_parquet(results_dir / "scores_anchor.parquet").set_index("item_id")
    sT = pd.read_parquet(results_dir / "scores_test.parquet").set_index("item_id")
    allS = pd.concat([sA, sT])
    feats = list(sA.columns)
    fw = allS.loc[mp.winner_id, feats].to_numpy()
    fl = allS.loc[mp.loser_id, feats].to_numpy()
    s = PT.assign_signs(mp, seed=seed)  # shared with PT.build_design -> McNemar-valid
    return (fw - fl) * s[:, None], feats


# --------------------------------------------------------------------------- #
# arm assembly (+ disk cache)
# --------------------------------------------------------------------------- #
def _text_arm(mp, kind, seed):
    """(X, names) for a vectorized text arm. Runs the vectorizer exactly once."""
    vecs, vocab = unit_vectors(mp, kind)
    X, _ = design_from_vecs(mp, vecs, seed=seed)
    return X, [f"{kind}:{t}" for t in vocab]


def _save(cache_dir, arm, X, names):
    p = Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    stem = arm.replace("/", "_").replace("#", "n").replace("+", "_plus_")
    if sparse.issparse(X):
        sparse.save_npz(p / f"{stem}.npz", X)
    else:
        np.save(p / f"{stem}.npy", X)
    (p / f"{stem}.names.json").write_text(json.dumps(names))


def _load(cache_dir, arm):
    p = Path(cache_dir)
    stem = arm.replace("/", "_").replace("#", "n").replace("+", "_plus_")
    npz, npy, nm = p / f"{stem}.npz", p / f"{stem}.npy", p / f"{stem}.names.json"
    if not nm.exists():
        return None
    if npz.exists():
        return sparse.load_npz(npz), json.loads(nm.read_text())
    if npy.exists():
        return np.load(npy), json.loads(nm.read_text())
    return None


def build_arm_designs(mp, arms, results_dir=None, cache_dir=None, seed=SEED):
    """Assemble the requested arms' designs. Returns ``(designs, y)``.

    ``designs`` maps arm name -> ``(X, feature_names)``. ``arms`` is a set of lowercase keys from
    ``{#words, interplay, interplay+pos, bow, pos, all-tan, bt}``; ``#words`` and ``interplay`` are
    always built (they are cheap and are the reference arms). ``bt`` requires ``results_dir``.

    When ``cache_dir`` is given, each design is memoized to disk -- the spaCy POS pass over ~8.4k
    documents and the BOW vectorization cost minutes and must not be repaid per k.
    """
    want = {a.strip().lower() for a in arms if a and a.strip()}
    designs = {}

    def cached(arm, build):
        if cache_dir is not None:
            hit = _load(cache_dir, arm)
            if hit is not None:
                designs[arm] = hit
                return
        X, names = build()
        if cache_dir is not None:
            _save(cache_dir, arm, X, names)
        designs[arm] = (X, names)

    # dense text designs (shared seed 42 -> identical sign/label -> McNemar-valid across arms)
    ex_w, nm_w = WF.block_extractor(["words"])
    ex_i, nm_i = WF.block_extractor(["interplay"])
    Xw, y = PT.build_design(mp, "pos_text", "neg_text", ex_w, op_col="op_text", seed=seed)
    Xi, _ = PT.build_design(mp, "pos_text", "neg_text", ex_i, op_col="op_text", seed=seed)
    designs["#words"] = (Xw, nm_w)
    designs["interplay"] = (Xi, nm_i)

    if "bt" in want:
        if results_dir is None:
            raise ValueError("arm 'bt' requires results_dir")
        Xbt, fbt = bt_design(mp, results_dir, seed=seed)
        designs[f"BT-{len(fbt)}"] = (Xbt, fbt)

    Xbow = Xpos = None
    nbow = npos = None
    if {"bow", "all-tan"} & want:
        cached("BOW", lambda: _text_arm(mp, "bow", seed))
        Xbow, nbow = designs["BOW"]
        if "bow" not in want:
            del designs["BOW"]
    if {"pos", "all-tan", "interplay+pos"} & want:
        cached("POS", lambda: _text_arm(mp, "pos", seed))
        Xpos, npos = designs["POS"]
        if "pos" not in want:
            del designs["POS"]

    if "interplay+pos" in want:
        designs["interplay+POS"] = (
            sparse.hstack([sparse.csr_matrix(Xi), Xpos], format="csr"),
            nm_i + npos,
        )
    if "all-tan" in want:
        # sparse hstack; the dense text blocks carry no NaN in practice (min_words>=20 guarantees a
        # non-empty reply, so the overlap ratios are defined) -> 0-fill == the mean-imputer no-op.
        assert not (np.isnan(Xw).any() or np.isnan(Xi).any()), "NaN in text design; imputer needed"
        blocks = [sparse.csr_matrix(Xw), sparse.csr_matrix(Xi)]
        names = nm_w + nm_i
        for b, n in ((Xbow, nbow), (Xpos, npos)):
            if b is not None:
                blocks.append(b)
                names = names + n
        designs["all-Tan"] = (sparse.hstack(blocks, format="csr"), names)

    return designs, y


# --------------------------------------------------------------------------- #
# uncertainty
# --------------------------------------------------------------------------- #
def boot_ci(correct, n_boot=4000, seed=0, pct=(2.5, 97.5)):
    """Percentile bootstrap CI over held-out pairs.

    The RNG is seeded afresh for every call, so a CI does not depend on the order in which arms
    are evaluated.
    """
    c = np.asarray(correct, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(c), size=(n_boot, len(c)))
    return tuple(np.percentile(c[idx].mean(axis=1), pct))
