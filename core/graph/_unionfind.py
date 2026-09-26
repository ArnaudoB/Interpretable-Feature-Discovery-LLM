"""Tiny union-find for the comparison-graph connectivity check.

Used for connectivity checks when building the nested comparison graph
(each prefix of the matching union must stay a single connected component).
"""

from __future__ import annotations

import numpy as np


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int64)

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return int(i)

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri == rj:
            return
        if self.rank[ri] < self.rank[rj]:
            ri, rj = rj, ri
        self.parent[rj] = ri
        if self.rank[ri] == self.rank[rj]:
            self.rank[ri] += 1

    def n_components(self) -> int:
        return len({self.find(i) for i in range(len(self.parent))})


def is_connected(edges: np.ndarray, n: int) -> bool:
    """``edges`` is an (E, 2) int array of (i, j) pairs; return True iff one component."""
    uf = UnionFind(n)
    for i, j in edges:
        uf.union(int(i), int(j))
    return uf.n_components() == 1
