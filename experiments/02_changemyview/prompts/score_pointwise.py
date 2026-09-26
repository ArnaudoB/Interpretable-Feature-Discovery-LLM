"""Pointwise scoring prompt for the CMV feature sets (Prompt `prm:cmv-score-pw`).

The judge rates ONE exchange on all K criteria in a single call, 0-10 each. No comparison,
no anchor graph, no Bradley-Terry: the score is read straight off the model's rating. Holding
the criteria fixed, the arm asks whether the pairwise apparatus earns its keep.

**This is deliberately a near-copy of the BT prompt (``prompts/score_bt.py``).** Framing
sentence, background, criteria rendering, the per-criterion evidence phrase, and the
out-of-scope list are kept wherever they still apply, so the contrast between arms is the
scoring *mode* and not incidental prompt wording. Three things necessarily differ:

* One exchange instead of two, so the pairwise-only lines are gone ("the two exchanges
  concern different posters...", "the two differ by design").
* A scale has to be defined. Pairwise never needs one -- "which shows more" is
  self-anchoring, which is precisely the property this arm gives up. The scale is anchored
  on the population of replies rather than on the item, because the scores must be
  comparable *across* exchanges scored in separate calls.
* An explicit push against central tendency, mirroring the pairwise prompt's "even when the
  two are close, pick the one that shows more". Without it, pointwise ratings pile up at 5-7
  and the arm loses to compression rather than to anything substantive.

The scale is 0-10 with no per-score anchor descriptors, which would make this a different,
richer arm rather than the plain pointwise counterpart of the pairwise prompt.

Outcomes are never mentioned.
"""

from __future__ import annotations

import json

SCALE_LO, SCALE_HI = 0, 10

POINTWISE_SYSTEM = (
    "You are helping apply an interpretable framework for what drives persuasion in debate. "
    "The framework scores an attempt to change someone's mind along a fixed set of "
    "dimensions, listed below.\n"
    "\n"
    'Background. Each item is an online discussion in which an original poster (the "OP") '
    "states a view they hold and invites others to change it; a challenger then replies with "
    "an argument aimed at changing that specific view. You will be shown one such exchange, "
    "presented as a block that begins with the OP's view (marked [OP VIEW]) followed by the "
    "challenger's reply (marked [CHALLENGE]).\n"
    "\n"
    "The dimensions:\n"
    "[[CRITERIA]]\n"
    "\n"
    "Your task. For EACH listed dimension, in order, give a short evidence phrase (the "
    "concrete feature you saw) and then rate the exchange on that dimension from "
    f"{SCALE_LO} to {SCALE_HI}. Some dimensions describe the challenger's argument, others "
    "describe the OP's post — assess each exactly as its definition states.\n"
    "\n"
    "What the numbers mean. Rate each dimension relative to online arguments of this kind in "
    f"general, not relative to this exchange's other dimensions: {SCALE_LO} means the "
    f"dimension is entirely absent, 5 means it is present to a typical degree, and "
    f"{SCALE_HI} means it is present about as strongly as such a reply ever gets. The same "
    "rating must mean the same thing on every exchange you score, because these scores are "
    "compared across exchanges you will not see together.\n"
    "\n"
    "Judge each dimension on its own terms: exchanges have uneven profiles, so it is entirely "
    "normal for one to score high on some dimensions and low on others. Use the full range — "
    "do not cluster every dimension in the middle.\n"
    "\n"
    "Out of scope — do not let these drive any rating: the topic or subject matter and how "
    "interesting, important, difficult, or agreeable the OP's view is; which side you "
    "personally agree with, or whether the OP's view is correct; usernames or identifiers; "
    "length in itself.\n"
    "\n"
    "Return one item per listed dimension, in the same order, naming the dimension exactly as "
    "given, each with a short evidence phrase and an integer rating.\n"
    "\n"
    "Output only JSON of the form:\n"
    '{"criteria": [{"name": "<dimension>", "evidence": "<short phrase>", '
    f'"score": <integer {SCALE_LO}-{SCALE_HI}>}}, ...]}}'
)

POINTWISE_USER = (
    "Exchange:\n{exchange}\n\n"
    "Following the standard above, go through each listed dimension in order: give the "
    "concrete evidence you saw and then its rating. Respond with the JSON object only."
)

POINTWISE_SCHEMA = {
    "name": "persuasion_criteria_pointwise",
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
                    # evidence before score so the phrase is generated first and conditions the number
                    "required": ["name", "evidence", "score"],
                    "properties": {
                        "name": {"type": "string"},
                        "evidence": {"type": "string"},
                        "score": {"type": "integer"},
                    },
                },
            }
        },
    },
}


def render_criteria(criteria) -> str:
    """``criteria`` = [{name, definition}] -> numbered 'name: definition' list.

    Identical to ``prompts/score_bt.py``'s ``render_criteria``, so both arms see identical
    criterion text.
    """
    return "\n".join(
        f"{i + 1}. {c['name']}: {c.get('definition', c.get('description', ''))}"
        for i, c in enumerate(criteria)
    )


def pointwise_messages(exchange: str, criteria) -> tuple[str, str]:
    """Return (system, user); criteria are baked into the (cacheable) system prompt."""
    system = POINTWISE_SYSTEM.replace("[[CRITERIA]]", render_criteria(criteria))
    return system, POINTWISE_USER.format(exchange=exchange)


def parse_pointwise(text: str, feature_names, lo: int = SCALE_LO, hi: int = SCALE_HI) -> dict:
    """Parse the JSON -> {feature_name: int score}, clamped to [lo, hi].

    Aligns by exact name, falling back to positional order when the returned count matches
    but names do not — the same tolerance ``prompts/score_bt.py``'s ``parse_specified`` applies.
    """
    t = text.strip()
    if t.startswith("```"):
        t = t[t.find("{") : t.rfind("}") + 1]
    items = json.loads(t).get("criteria", [])
    by_name = {}
    for it in items:
        nm, sc = it.get("name"), it.get("score")
        if isinstance(nm, str) and isinstance(sc, (int, float)):
            by_name[nm.strip()] = int(max(lo, min(hi, round(float(sc)))))
    out = {f: by_name[f] for f in feature_names if f in by_name}
    if not out and len(items) == len(feature_names):
        for f, it in zip(feature_names, items):
            sc = it.get("score")
            if isinstance(sc, (int, float)):
                out[f] = int(max(lo, min(hi, round(float(sc)))))
    return out
