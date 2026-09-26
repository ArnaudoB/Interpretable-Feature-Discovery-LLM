"""OpenAI sentence-embedding feature blocks for the CMV pairwise task (``emb`` / ``emb-sim``).

Both arms are per-``(reply, OP)`` extractors with the signature the rest of this package
expects -- ``f(unit_text, op_text) -> list[float]`` -- so they feed straight through
:func:`core.cmv.pairtask.build_design` and inherit the shared seed-42 sign deck.
That is what keeps them paired-McNemar-comparable with the Tan arms and the BT panel.

``emb``
    The whole-reply embedding (3,072 dims for ``text-embedding-3-large``). The dense
    counterpart of BOW: an opaque, high-dimensional text representation. The OP
    embedding is deliberately **not** concatenated -- Tan's matching gives both rows of
    a pair the same poster, so any OP-only feature cancels exactly in the within-pair
    difference.

``emb-sim``
    Eleven similarity statistics between reply and OP. This is the dense counterpart of
    :func:`core.cmv.features.interplay_features`: interplay measures word-set
    overlap between the reply and the OP, both whole-text and over a 4x4 grid of text
    quarters; ``emb-sim`` measures cosine similarity over those same two structures,
    plus a finer sentence-chunk coverage matrix that recovers the *asymmetry* interplay
    gets from ``reply_frac`` vs ``op_frac`` (cosine itself is symmetric).

    Deliberately excluded: chunk counts and any length term. Interplay carries no length
    feature either -- ``#words`` is its own arm -- so including one here would make the
    two arms non-comparable. The coverage means retain a mild chunk-count dependence,
    which is what the shared ``#words`` residualization pass exists to strip.

**Structural limitation, stated up front.** Interplay's headline finding rests on
splitting overlap into stopword vs content-word sets, and sentence encoders discard
function-word style almost entirely. There is no analogue of ``features._split_wordsets``
here and there cannot be one. If interplay's signal lives in that contrast, ``emb-sim``
will underperform it, and that is the result rather than a defect of this implementation.

All vectors are L2-normalized by the API, so cosine similarity is a plain dot product.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

#: USD per 1M input tokens. ESTIMATE, in the spirit of ``core.judging.pricing`` -- but the
#: embeddings endpoint returns real usage, so callers should report actual tokens instead.
PRICE_PER_1M = 0.13

DEFAULT_MODEL = "text-embedding-3-large"
DEFAULT_DIM = 3072

# --- chunking spec: fixed before any heldout accuracy was computed -------------------- #
#: Greedy sentence packing target. Chunks never split a sentence unless the sentence
#: alone exceeds this, in which case it is hard-split on word boundaries.
CHUNK_WORDS = 60
#: Cap on chunks per document. Bounds cost on the long tail (the longest OP is ~3.6k
#: words -> 61 chunks), and only binds for a handful of posts.
MAX_CHUNKS = 64
#: Defensive per-text truncation. text-embedding-3-large accepts 8,191 tokens; the
#: longest document here is well under that, so this effectively never fires.
MAX_CHARS = 30_000

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

EMB_SIM_NAMES = [
    "cos_whole",  # cosine(reply, OP) on whole documents
    "q_max",  # 4x4 quarter-grid cosine: max  (analogue of interplay q_*_max)
    "q_min",  # 4x4 quarter-grid cosine: min  (analogue of interplay q_*_min)
    "q_mean",  # 4x4 quarter-grid cosine: mean
    "reply_cov",  # mean over reply chunks of best-matching OP chunk ~ interplay reply_frac
    "op_cov",  # mean over OP chunks of best-matching reply chunk ~ interplay op_frac
    "s_max",  # chunk cosine matrix: max
    "s_min",  # chunk cosine matrix: min
    "s_mean",  # chunk cosine matrix: mean
    "s_std",  # chunk cosine matrix: sd (spread of engagement)
    "reply_coherence",  # mean off-diagonal cosine among the reply's own chunks
]


# --------------------------------------------------------------------------- #
# text decomposition
# --------------------------------------------------------------------------- #
def quarter_texts(text: str) -> list:
    """Four contiguous word-quarters, mirroring :func:`features._quarters`.

    Interplay builds its 4x4 grid from token quarters; this rejoins the same quarters
    into strings so they can be embedded. Whitespace is normalized in the process, which
    is immaterial to a sentence encoder. Texts with fewer than 4 words fall back to four
    copies of the whole text (the corpus filter guarantees >= 20 words, so this is a
    guard, not a code path that runs).
    """
    words = text.split()
    if len(words) < 4:
        whole = " ".join(words) or "."
        return [whole] * 4
    step = len(words) / 4.0
    return [" ".join(words[int(i * step) : int((i + 1) * step)]) or "." for i in range(4)]


def sentence_chunks(text: str, max_words: int = CHUNK_WORDS, max_chunks: int = MAX_CHUNKS) -> list:
    """Greedily pack sentences into chunks of at most ``max_words`` words.

    Sentences are kept intact unless one alone exceeds the budget, in which case it is
    hard-split on word boundaries. Always returns at least one non-empty chunk.
    """
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    if not sents:
        sents = [text.strip() or "."]

    pieces: list = []
    for s in sents:
        w = s.split()
        if len(w) <= max_words:
            pieces.append(w)
            continue
        for i in range(0, len(w), max_words):
            pieces.append(w[i : i + max_words])

    chunks: list = []
    cur: list = []
    for w in pieces:
        if cur and len(cur) + len(w) > max_words:
            chunks.append(" ".join(cur))
            cur = []
        cur.extend(w)
    if cur:
        chunks.append(" ".join(cur))
    return chunks[:max_chunks] or ["."]


def required_texts(mp) -> list:
    """Every distinct string that must be embedded to build both arms for ``mp``.

    Order-preserving dedup: replies and OPs recur (a pair's two rows share an OP, and
    the same OP backs several pairs), so this is materially smaller than the naive count.
    """
    seen: dict = {}
    for col in ("pos_text", "neg_text", "op_text"):
        for t in mp[col]:
            for s in (t, *quarter_texts(t), *sentence_chunks(t)):
                if s not in seen:
                    seen[s] = None
    return list(seen)


# --------------------------------------------------------------------------- #
# vector store
# --------------------------------------------------------------------------- #
class EmbeddingStore:
    """Append-only disk cache: one flat float32 blob plus a sha256 -> row index.

    There are ~120k vectors of 3,072 floats, so per-vector files would burn 120k inodes; a
    single blob keeps it to two files.

    Crash safety: the index is written after the blob, and :meth:`_open` truncates any
    trailing vectors not covered by the index, so an interrupted run resumes cleanly
    instead of silently shifting every subsequent row.
    """

    def __init__(self, cache_dir, model: str = DEFAULT_MODEL, dim: int = DEFAULT_DIM):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.model, self.dim = model, dim
        self.blob = self.dir / "vectors.f32"
        self.index_path = self.dir / "index.json"
        self.meta_path = self.dir / "meta.json"
        self._open()

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}::{text}".encode("utf-8")).hexdigest()

    def _open(self):
        if self.meta_path.exists():
            meta = json.loads(self.meta_path.read_text())
            if meta["model"] != self.model or meta["dim"] != self.dim:
                raise ValueError(
                    f"cache at {self.dir} holds {meta['model']}/{meta['dim']}d, "
                    f"asked for {self.model}/{self.dim}d — use a different cache dir"
                )
        else:
            self.meta_path.write_text(json.dumps({"model": self.model, "dim": self.dim}))
        self.index = json.loads(self.index_path.read_text()) if self.index_path.exists() else {}
        want_bytes = len(self.index) * self.dim * 4
        have_bytes = self.blob.stat().st_size if self.blob.exists() else 0
        if have_bytes > want_bytes:
            log.warning(
                "truncating %d orphan bytes from %s (interrupted run)",
                have_bytes - want_bytes,
                self.blob,
            )
            with open(self.blob, "r+b") as fh:
                fh.truncate(want_bytes)
        elif have_bytes < want_bytes:
            raise ValueError(f"{self.blob} is shorter than its index claims — cache corrupt")

    def missing(self, texts: list) -> list:
        """Subset of ``texts`` with no cached vector, order-preserving and deduplicated."""
        out, seen = [], set()
        for t in texts:
            k = self._key(t)
            if k not in self.index and k not in seen:
                seen.add(k)
                out.append(t)
        return out

    def add(self, texts: list, vecs: np.ndarray):
        """Append vectors for ``texts``. Blob first, then index (see crash safety)."""
        assert len(texts) == len(vecs), "texts/vectors length mismatch"
        with open(self.blob, "ab") as fh:
            fh.write(np.ascontiguousarray(vecs, dtype=np.float32).tobytes())
        n = len(self.index)
        for i, t in enumerate(texts):
            self.index[self._key(t)] = n + i
        self.index_path.write_text(json.dumps(self.index))

    def matrix(self) -> np.ndarray:
        """The whole blob as an ``(n_cached, dim)`` read-only memmap."""
        return np.memmap(self.blob, dtype=np.float32, mode="r").reshape(-1, self.dim)

    def lookup(self) -> callable:
        """``text -> vector`` closure over an in-memory copy of the blob.

        Materialized rather than memmapped: the extractors touch every row many times
        (each pair reads its OP's chunks twice), and page-faulting a 1.4 GB file per
        access is far slower than holding it.
        """
        M = np.array(self.matrix())
        idx = self.index

        def vec(text: str) -> np.ndarray:
            return M[idx[self._key(text)]]

        return vec


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
def _batches(texts: list, max_items: int = 256, max_chars: int = 480_000):
    """Split into request-sized batches. ``max_chars`` ~ 120k tokens, well inside the
    endpoint's per-request cap; ``max_items`` is far under the 2,048 array limit."""
    cur, cur_chars = [], 0
    for t in texts:
        c = min(len(t), MAX_CHARS)
        if cur and (len(cur) >= max_items or cur_chars + c > max_chars):
            yield cur
            cur, cur_chars = [], 0
        cur.append(t)
        cur_chars += c
    if cur:
        yield cur


def embed_missing(
    store: EmbeddingStore, texts: list, client, workers: int = 8, progress=None
) -> dict:
    """Embed everything in ``texts`` not already cached, writing into ``store``.

    Returns a usage dict ``{n_texts, n_requests, prompt_tokens, usd}``. Batches are
    embedded concurrently but committed to the store serially on the main thread, so
    the append-only blob never interleaves.
    """
    from core.judging.retry import with_backoff

    todo = store.missing(texts)
    if not todo:
        return {"n_texts": 0, "n_requests": 0, "prompt_tokens": 0, "usd": 0.0}

    batches = list(_batches(todo))
    log.info("embedding %d new texts in %d requests", len(todo), len(batches))

    def run(batch):
        payload = [t[:MAX_CHARS] for t in batch]
        resp = with_backoff(client.embeddings.create, model=store.model, input=payload)
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        assert vecs.shape == (len(batch), store.dim), f"unexpected shape {vecs.shape}"
        return batch, vecs, int(resp.usage.prompt_tokens)

    total_tok = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (batch, vecs, tok) in enumerate(ex.map(run, batches), 1):
            store.add(batch, vecs)
            total_tok += tok
            if progress and (i % 10 == 0 or i == len(batches)):
                progress(i, len(batches), total_tok)

    return {
        "n_texts": len(todo),
        "n_requests": len(batches),
        "prompt_tokens": total_tok,
        "usd": total_tok / 1e6 * PRICE_PER_1M,
    }


# --------------------------------------------------------------------------- #
# feature blocks
# --------------------------------------------------------------------------- #
def _mean_offdiag(G: np.ndarray) -> float:
    """Mean of the strict upper triangle; 0.0 when there is no off-diagonal cell."""
    n = G.shape[0]
    if n < 2:
        return 0.0
    iu = np.triu_indices(n, k=1)
    return float(G[iu].mean())


class EmbExtractor:
    """``emb`` arm: the whole-reply embedding. ``op_text`` is accepted and ignored."""

    def __init__(self, vec):
        self.vec = vec

    def __call__(self, unit_text: str, op_text: str) -> list:
        return self.vec(unit_text).astype(float).tolist()


class EmbSimExtractor:
    """``emb-sim`` arm: the 11 reply-vs-OP similarity statistics in :data:`EMB_SIM_NAMES`."""

    def __init__(self, vec):
        self.vec = vec

    def _stack(self, texts: list) -> np.ndarray:
        return np.stack([self.vec(t) for t in texts])

    def __call__(self, unit_text: str, op_text: str) -> list:
        vec = self.vec
        r_w, o_w = vec(unit_text), vec(op_text)

        # 4x4 quarter grid — the structural analogue of interplay's qmax/qmin block.
        G = self._stack(quarter_texts(unit_text)) @ self._stack(quarter_texts(op_text)).T

        # sentence-chunk matrix — finer, and asymmetric via the two coverage means.
        rc = self._stack(sentence_chunks(unit_text))
        oc = self._stack(sentence_chunks(op_text))
        S = rc @ oc.T

        return [
            float(r_w @ o_w),
            float(G.max()),
            float(G.min()),
            float(G.mean()),
            float(S.max(axis=1).mean()),  # reply grounded in the OP     ~ reply_frac
            float(S.max(axis=0).mean()),  # OP addressed by the reply    ~ op_frac
            float(S.max()),
            float(S.min()),
            float(S.mean()),
            float(S.std()),
            _mean_offdiag(rc @ rc.T),
        ]
