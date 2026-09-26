"""text-embedding-3-large with a content-addressed on-disk cache.

Used to place elicited criteria and the engineered dials in one embedding space so
they can be matched (Hungarian) and the recovery metrics computed. Vectors are
L2-normalized, so cosine similarity is a plain dot product. After the first call the
cache makes all downstream analysis offline and deterministic.

The API key is read from ``OPENAI_API_KEY``, falling back to the repository-root ``.env``
(see :mod:`core.openai_client`); no key is needed when every string is already cached.

``DEFAULT_CACHE`` (``.embed_cache``) is relative to the *current working directory*. The
experiment scripts pass an explicit ``cache_dir`` (e.g. ``<experiment>/.embed_cache``,
which in Exp 1 is a symlink to the shipped ``raw/embed_cache``); calls relying on the
default must be run from the experiment directory to hit the shipped cache.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

EMB_MODEL = "text-embedding-3-large"
EMB_DIM = 3072
DEFAULT_CACHE = Path(".embed_cache")

_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        from core.openai_client import get_client

        _CLIENT = get_client()
    return _CLIENT


def _key(text: str) -> str:
    return hashlib.sha256(f"{EMB_MODEL}:{EMB_DIM}:{text}".encode()).hexdigest()


def embed_texts(texts, *, cache_dir: Path = DEFAULT_CACHE, batch: int = 256) -> np.ndarray:
    """L2-normalized text-embedding-3-large vectors for ``texts`` (cosine = dot product).

    Each unique string is cached to ``cache_dir/<sha256>.npy``; only cache misses hit
    the API. Returns an array aligned to the input order (duplicates share a vector).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    uniq = list(dict.fromkeys(texts))
    out: dict[str, np.ndarray] = {}
    for t in uniq:
        p = cache_dir / f"{_key(t)}.npy"
        if p.exists():
            out[t] = np.load(p)
    miss = [t for t in uniq if t not in out]
    for i in range(0, len(miss), batch):
        chunk = miss[i : i + batch]
        resp = _client().embeddings.create(model=EMB_MODEL, input=chunk, dimensions=EMB_DIM)
        for t, d in zip(chunk, resp.data):
            v = np.asarray(d.embedding, np.float32)
            v /= np.linalg.norm(v) + 1e-12
            np.save(cache_dir / f"{_key(t)}.npy", v)
            out[t] = v
    return np.array([out[t] for t in texts])


def cosine_matrix(row_texts, col_texts, *, cache_dir: Path = DEFAULT_CACHE) -> np.ndarray:
    """(len(row_texts) x len(col_texts)) cosine-similarity matrix in embedding space."""
    R = embed_texts(list(row_texts), cache_dir=cache_dir)
    C = embed_texts(list(col_texts), cache_dir=cache_dir)
    return R @ C.T
