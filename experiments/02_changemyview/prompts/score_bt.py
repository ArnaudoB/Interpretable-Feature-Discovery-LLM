"""Specified-criteria comparative prompt for per-feature BT scoring (Prompt `prm:cmv-score-bt`).

The judge is handed a FIXED list of criteria (name + definition) and, for EACH one, gives a short
evidence phrase and a forced winner ('A' or 'B') -- no "differs" gate and no overall verdict, so
every feature yields an outcome on every pair (plain Bradley-Terry). Persuasion-domain,
de-identified (no platform named). The schema has a per-criterion ``evidence`` field and no
``overall_winner``.
"""

from __future__ import annotations

import json

SPECIFIED_SYSTEM = (
    "You are helping apply an interpretable framework for what drives persuasion in debate. "
    "The framework scores how two attempts to change someone's mind differ along a fixed set of "
    "dimensions, listed below.\n"
    "\n"
    'Background. Each item is an online discussion in which an original poster (the "OP") states '
    "a view they hold and invites others to change it; a challenger then replies with an argument "
    "aimed at changing that specific view. You will be shown two such exchanges, Exchange A and "
    "Exchange B, each presented as a block that begins with the OP's view (marked [OP VIEW]) "
    "followed by the challenger's reply (marked [CHALLENGE]). The two exchanges concern different "
    "posters holding different views on different topics.\n"
    "\n"
    "The dimensions:\n"
    "[[CRITERIA]]\n"
    "\n"
    "Your task. For EACH listed dimension, in order, give a short evidence phrase (the concrete "
    "feature you saw) and then indicate which exchange shows more of it — 'A' or 'B'. Some dimensions "
    "describe the challenger's argument, others describe the OP's post — assess each exactly as its "
    "definition states. Judge each dimension on its own terms: exchanges have uneven profiles, so it "
    "is entirely normal for A to show more of some dimensions and B more of others. Even when the "
    "two are close on a dimension, pick the exchange that shows more of it.\n"
    "\n"
    "Out of scope — do not let these drive any judgment: the topic or subject matter and how "
    "interesting, important, difficult, or agreeable either OP's view is (the two differ by design); "
    "which side you personally agree with, or whether an OP's view is correct; usernames or "
    "identifiers; length in itself.\n"
    "\n"
    "Return one item per listed dimension, in the same order, naming the dimension exactly as given, "
    "each with a short evidence phrase and the winner ('A' or 'B').\n"
    "\n"
    "Output only JSON of the form:\n"
    '{"criteria": [{"name": "<dimension>", "evidence": "<short phrase>", '
    '"winner": "A" | "B"}, ...]}'
)

SPECIFIED_USER = (
    "Exchange A:\n{exchange_a}\n\n"
    "Exchange B:\n{exchange_b}\n\n"
    "Following the standard above, go through each listed dimension in order: give the concrete "
    "evidence you saw and then say which exchange shows more of it. Compare on the listed dimensions, "
    "not on topic. Respond with the JSON object only."
)

SPECIFIED_SCHEMA = {
    "name": "persuasion_criteria",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["criteria"],
        "properties": {
            "criteria": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "evidence", "winner"],
                    "properties": {
                        "name": {"type": "string"},
                        "evidence": {"type": "string"},
                        "winner": {"type": "string", "enum": ["A", "B"]},
                    },
                },
            }
        },
    },
}


def render_criteria(criteria) -> str:
    """``criteria`` = [{name, definition}] -> numbered 'name: definition' list."""
    return "\n".join(
        f"{i + 1}. {c['name']}: {c.get('definition', c.get('description', ''))}"
        for i, c in enumerate(criteria)
    )


def specified_messages(exchange_a: str, exchange_b: str, criteria) -> tuple[str, str]:
    """Return (system, user); the 13 criteria are baked into the (cacheable) system prompt."""
    system = SPECIFIED_SYSTEM.replace("[[CRITERIA]]", render_criteria(criteria))
    user = SPECIFIED_USER.format(exchange_a=exchange_a, exchange_b=exchange_b)
    return system, user


def parse_specified(text: str, feature_names) -> dict:
    """Parse the JSON → {feature_name: winner('A'|'B')} for the known features.

    Robust to code fences and to missing/extra items; aligns by exact name, falling back to
    positional order when the returned count matches and names don't line up.
    """
    t = text.strip()
    if t.startswith("```"):
        t = t[t.find("{") : t.rfind("}") + 1]
    obj = json.loads(t)
    items = obj.get("criteria", [])
    by_name = {}
    for it in items:
        nm, w = it.get("name"), it.get("winner")
        if isinstance(nm, str) and w in ("A", "B"):
            by_name[nm.strip()] = w
    out = {}
    for f in feature_names:
        if f in by_name:
            out[f] = by_name[f]
    # positional fallback if names didn't match but the count lines up
    if not out and len(items) == len(feature_names):
        for f, it in zip(feature_names, items):
            if it.get("winner") in ("A", "B"):
                out[f] = it["winner"]
    return out
