"""Canonical normalization of rationale strings.

Judges name the same construct in many surface forms, so a dimension string is
lowercased, stripped of punctuation, and freed of leading determiners and of the
comparatives ``more`` and ``better``: ``"more coherent argument"`` and
``"coherent argument"`` collapse to the same normalized form before embedding.
Stripping is applied repeatedly until no leading determiner / comparative
remains.
"""

from __future__ import annotations

import re

_LEADING = re.compile(r"^(?:the|a|an|more|better)\s+", re.IGNORECASE)
_WS = re.compile(r"\s+")


def normalize_quality(s: str) -> str:
    """Return a normalized canonical form of ``s``.

    Examples:
        >>> normalize_quality("The clearer thesis")
        'clearer thesis'
        >>> normalize_quality("  A More Coherent argument  ")
        'coherent argument'
        >>> normalize_quality("Better grammar")
        'grammar'
        >>> normalize_quality("more effective use of transitions")
        'effective use of transitions'
        >>> normalize_quality("an unambiguous claim")
        'unambiguous claim'
    """
    s = (s or "").strip().lower()
    # Repeatedly strip leading determiners/comparatives ("a more clear" -> "clear").
    while True:
        new = _LEADING.sub("", s)
        if new == s:
            break
        s = new
    s = _WS.sub(" ", s)
    return s.strip()
