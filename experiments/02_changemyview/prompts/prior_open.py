"""Prior-only feature elicitation, OPEN-COUNT variant -- "name as many as the task warrants".

The counterpart of ``prompts/prior.py`` (the exactly-16 P set) that differs only in the count:
the model decides how many dimensions to name, and is asked to be thorough. Everything the
16-feature prompt leaves out is still left out -- in particular no families of dimension are
named, so breadth is pushed on without seeding content. The per-draw counts of its two runs of
10 draws are the open-count figures of App. `app:methods-cmv` (cell ``feature_prior_root_open``).

Why no target number: naming one (e.g. "about 40") makes the model pad the list to hit it, so
the count would measure the instruction rather than the model's prior. The anti-padding clause
in the task paragraph is the counterweight to "be thorough".

The breadth instruction talks only about the LIST, never about the data: a phrase such as
dimensions "that show up in only some replies" would be incoherent here (this model sees no
replies) and an empirical claim, for the same reason the base rate and the corpus description
are absent.

See ``prompts/prior.py`` for what is deliberately absent from both prompts and why.
"""

from __future__ import annotations

import json

N_FEATURES = None  # open count: the model decides

#: Primary — the setting only, no corpus named.
BACKGROUND = (
    "In online debates, a person states a view they hold and invites others to change it. "
    "Others reply with arguments aimed at changing that view."
)

#: Contamination contrast — names the corpus. Diagnostic only, never the reported baseline:
#: a materially better panel here means the model is recalling a literature rather than
#: reasoning from priors, which the existing contamination probe can corroborate.
NAMED_BACKGROUND = (
    "The data is the ChangeMyView corpus of Tan et al. (2016), collected from the "
    "/r/ChangeMyView subreddit: an original poster states a view and invites "
    "counterarguments, and awards a delta to any reply that changed their view."
)

PRIOR_SYSTEM = """You are helping build a framework for assessing how persuasive an argument is.

[[BACKGROUND]]

Your task. Name the dimensions for assessing how persuasive such a reply is. Give as many as the task warrants — decide the number yourself, and be thorough rather than selective. Do not pad the list either: add a dimension only when it names something none of the others already capture.

Each dimension will be used as a measurement scale: a judge will be shown two replies and asked which of the two shows more of it. So each dimension must be a matter of degree rather than a yes/no fact, and they must be distinct from one another.

For each dimension give:
- "name": a short label, 2-5 words;
- "definition": one sentence saying what it measures.

Output only JSON of the form:
{"criteria": [{"name": "<short label>", "definition": "<one sentence>"}, ...]}"""

PRIOR_USER = "Name the dimensions now, as a single JSON object. Respond with the JSON only."

PRIOR_SCHEMA = {
    "name": "prior_persuasion_criteria",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["criteria"],
        "properties": {
            "criteria": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "definition"],
                    "properties": {"name": {"type": "string"}, "definition": {"type": "string"}},
                },
            }
        },
    },
}


def prior_messages(n=N_FEATURES, named: bool = False) -> tuple[str, str]:
    """Return ``(system, user)``. ``n`` is accepted for signature parity and ignored (open count);
    ``named=True`` selects the contamination-contrast variant."""

    def fill(t: str) -> str:
        return t.replace("[[BACKGROUND]]", NAMED_BACKGROUND if named else BACKGROUND)

    return fill(PRIOR_SYSTEM), fill(PRIOR_USER)


MIN_SANE = 5  # a shorter list means the model misread the task, not that it was terse


def parse_prior(text: str, n=N_FEATURES) -> list:
    """Parse the JSON into ``[{name, definition}]``, the shape ``features.json`` expects.

    ``n`` is ignored when None (open count); only the sanity floor applies."""
    t = text.strip()
    if t.startswith("```"):
        t = t[t.find("{") : t.rfind("}") + 1]
    items = json.loads(t)["criteria"]
    out = []
    for it in items:
        nm, df = str(it["name"]).strip(), str(it["definition"]).strip()
        if nm and df:
            out.append({"name": nm, "definition": df})
    if n is not None and len(out) != n:
        raise ValueError(f"expected {n} criteria, parsed {len(out)}")
    if len(out) < MIN_SANE:
        raise ValueError(f"only {len(out)} criteria parsed (floor {MIN_SANE}); check the response")
    return out


def to_features_json(criteria: list, source: str) -> dict:
    """Wrap a parsed panel in the ``features.json`` envelope the BT pipeline reads."""
    return {"n_features": len(criteria), "source": source, "criteria": criteria}
