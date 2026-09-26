"""Nested balanced comparison graph: a union of random perfect matchings whose
every prefix is a connected regular subgraph, so degrees [5, 10, ..., 40] give a
pairs-scaling curve.
"""

from core.graph.matching import build_nested_pairs, verify_budgets

__all__ = ["build_nested_pairs", "verify_budgets"]
