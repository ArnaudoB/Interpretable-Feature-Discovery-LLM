"""Independently-togglable feature blocks for the pairwise persuasion task.

Each block is a function ``f(unit_text, op_text) -> list[float]`` returning a
fixed-length vector, plus a matching ``*_names()`` for interpretability. Blocks
implemented: ``words`` (baseline) and ``interplay`` (argument/OP overlap).

`interplay` is the paper's central feature set: word-overlap between the argument
A and the original post O, on **unique words** (set, not multiset) semantics,
using **Mallet's** stoplist to split stopwords from content words.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

#: The MALLET English stoplist ships next to this module (``mallet_en_stoplist.txt``).
_LEX_DIR = Path(__file__).resolve().parent
_WORD_RE = re.compile(r"[a-z]+")


@lru_cache(maxsize=1)
def mallet_stopwords() -> frozenset:
    """The vendored Mallet English stoplist (524 words) as a lowercased set."""
    path = _LEX_DIR / "mallet_en_stoplist.txt"
    return frozenset(w.strip().lower() for w in path.read_text().splitlines() if w.strip())


def word_tokens(text: str) -> list:
    """Lowercased alphabetic tokens (for the interplay word-set semantics)."""
    return _WORD_RE.findall(text.lower())


# ----------------------------------------------------------------------------- #
# words (baseline)
# ----------------------------------------------------------------------------- #
def words_features(unit_text: str, op_text: str) -> list:
    return [float(len(unit_text.split()))]


def words_names() -> list:
    return ["n_words"]


# ----------------------------------------------------------------------------- #
# interplay
# ----------------------------------------------------------------------------- #
_WORDSETS = ("stop", "content", "all")
_METRICS = ("common", "reply_frac", "op_frac", "jaccard")


def _overlap(sa: set, so: set) -> list:
    """[|A∩O|, |A∩O|/|A|, |A∩O|/|O|, |A∩O|/|A∪O|] with 0 for empty denominators."""
    common = len(sa & so)
    na, no_, union = len(sa), len(so), len(sa | so)
    return [
        float(common),
        common / na if na else 0.0,
        common / no_ if no_ else 0.0,
        common / union if union else 0.0,
    ]


def _split_wordsets(tokens: list) -> dict:
    sw = mallet_stopwords()
    all_ = set(tokens)
    stop = {t for t in all_ if t in sw}
    return {"stop": stop, "content": all_ - stop, "all": all_}


def _wordset_metrics(a_tokens: list, o_tokens: list) -> list:
    """12 metrics = 4 overlap metrics x 3 word sets (stop, content, all)."""
    a, o = _split_wordsets(a_tokens), _split_wordsets(o_tokens)
    out: list = []
    for ws in _WORDSETS:
        out += _overlap(a[ws], o[ws])
    return out


def _quarters(tokens: list) -> list:
    """Four contiguous token quarters (paper §4.2.3 / Fig. 7)."""
    n = len(tokens)
    if n == 0:
        return [[], [], [], []]
    step = n / 4.0
    return [tokens[int(i * step) : int((i + 1) * step)] for i in range(4)]


def interplay_features(unit_text: str, op_text: str) -> list:
    """36 features: 12 whole-unit metrics + per-metric max & min over the 4x4
    quarter grid (order-independent quarter summary)."""
    at, ot = word_tokens(unit_text), word_tokens(op_text)
    whole = _wordset_metrics(at, ot)  # 12
    aq, oq = _quarters(at), _quarters(ot)
    grid = [_wordset_metrics(aq[i], oq[j]) for i in range(4) for j in range(4)]  # 16x12
    cols = list(zip(*grid))  # 12 columns of 16
    qmax = [max(c) for c in cols]
    qmin = [min(c) for c in cols]
    return whole + qmax + qmin  # 36


def interplay_names() -> list:
    whole = [f"{ws}_{m}" for ws in _WORDSETS for m in _METRICS]
    return (
        whole
        + [f"q_{ws}_{m}_max" for ws in _WORDSETS for m in _METRICS]
        + [f"q_{ws}_{m}_min" for ws in _WORDSETS for m in _METRICS]
    )


# ----------------------------------------------------------------------------- #
# registry
# ----------------------------------------------------------------------------- #
FEATURE_BLOCKS = {
    "words": (words_features, words_names),
    "interplay": (interplay_features, interplay_names),
}


def block_extractor(names: list):
    """Compose one or more blocks into a single ``f(unit, op) -> list`` + names."""
    fns = [FEATURE_BLOCKS[n][0] for n in names]
    colnames: list = []
    for n in names:
        colnames += [f"{n}:{c}" for c in FEATURE_BLOCKS[n][1]()]

    def extract(unit_text: str, op_text: str) -> list:
        out: list = []
        for fn in fns:
            out += fn(unit_text, op_text)
        return out

    return extract, colnames
