"""The two corpora, and the band-balanced sample every run is built on.

* :data:`ELLIPSE_TRAIT_COLS` — the six human analytic-trait columns of ELLIPSE.
* :func:`load_candidates` — read the shipped ELLIPSE CSV into the candidate pool.
* :func:`load_asap2_candidates` — the same schema from the ASAP 2.0 subset (no traits).
* :func:`select_balanced_essays` — the deterministic n=100 selection.
* :func:`assert_graph_invariants` — the hard checks the comparison graph must pass.
"""

from dataset.asap2 import load_asap2_candidates  # noqa: F401
from dataset.ellipse import (  # noqa: F401
    ELLIPSE_TRAIT_COLS,
    assert_graph_invariants,
    load_candidates,
    select_balanced_essays,
)

__all__ = [
    "ELLIPSE_TRAIT_COLS",
    "load_candidates",
    "load_asap2_candidates",
    "select_balanced_essays",
    "assert_graph_invariants",
]
