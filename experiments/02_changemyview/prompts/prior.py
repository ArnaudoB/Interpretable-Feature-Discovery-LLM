"""Prior-only feature elicitation, the P feature set (Prompt `prm:cmv-prior-elicit`).

The model is asked to name 16 dimensions for assessing how persuasive a reply is. It sees
**no exchanges, no outcomes, no statistics, and no description of the corpus** -- only the
one-line setting. The resulting set is then scored by the same BT and pointwise protocols as
the discovered features, so the only thing separating the P arms from the D arms is **where
the criteria came from**: one call to the model's prior, versus discovery from 150 exchanges.

**The prompt is deliberately thin.** Anything describing what *kinds* of dimensions to look for
pre-loads the answer: naming families like argument structure, evidence, style, or engagement
with the poster would hand over roughly the family structure the discovered set recovered, and
the arm would measure the hint rather than the model's prior.

Also deliberately absent:

* **Base rates.** No "most replies fail". The success rate is an empirical fact about the
  data that this arm is not allowed to know.
* **A scoring out-of-scope list.** The scoring prompts tell the judge to ignore topic,
  agreement, identifiers and length -- because that judge is looking at exchanges. This
  model sees no data, so the same list would carry no instruction, only the shape of the
  evaluation design.
* **Length constraints.** The discovered set was not elicited under a no-length rule, so
  imposing one here would hold this baseline to a stricter standard.

What remains beyond the ask itself is only what the output must satisfy to be *usable* by the
downstream scorer -- a graded scale rather than a yes/no fact, and 16 distinct entries. Those
are properties the discovered set has by construction (comparative elicitation is inherently
graded; canonicalization plus the reliability screen removes redundancy), so requiring them
matches the two sets' form rather than seeding their content.
"""

from __future__ import annotations

import json

N_FEATURES = 16

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

Your task. Name exactly [[N]] dimensions for assessing how persuasive such a reply is.

Each dimension will be used as a measurement scale: a judge will be shown two replies and asked which of the two shows more of it. So each dimension must be a matter of degree rather than a yes/no fact, and the [[N]] must be distinct from one another.

For each dimension give:
- "name": a short label, 2-5 words;
- "definition": one sentence saying what it measures.

Output only JSON of the form:
{"criteria": [{"name": "<short label>", "definition": "<one sentence>"}, ...]}"""

PRIOR_USER = "Name the [[N]] dimensions now, as a single JSON object. Respond with the JSON only."

PRIOR_SCHEMA = {
    "name": "prior_persuasion_criteria",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["criteria"],
        "properties": {
            "criteria": {
                "type": "array",
                "minItems": N_FEATURES,
                "maxItems": N_FEATURES,
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


def prior_messages(n: int = N_FEATURES, named: bool = False) -> tuple[str, str]:
    """Return ``(system, user)``. ``named=True`` selects the contamination-contrast variant."""

    def fill(t: str) -> str:
        return t.replace("[[BACKGROUND]]", NAMED_BACKGROUND if named else BACKGROUND).replace(
            "[[N]]", str(n)
        )

    return fill(PRIOR_SYSTEM), fill(PRIOR_USER)


def parse_prior(text: str, n: int = N_FEATURES) -> list:
    """Parse the JSON into ``[{name, definition}]``, the shape ``features.json`` expects."""
    t = text.strip()
    if t.startswith("```"):
        t = t[t.find("{") : t.rfind("}") + 1]
    items = json.loads(t)["criteria"]
    out = []
    for it in items:
        nm, df = str(it["name"]).strip(), str(it["definition"]).strip()
        if nm and df:
            out.append({"name": nm, "definition": df})
    if len(out) != n:
        raise ValueError(f"expected {n} criteria, parsed {len(out)}")
    return out


def to_features_json(criteria: list, source: str) -> dict:
    """Wrap a parsed panel in the ``features.json`` envelope the BT pipeline reads."""
    return {"n_features": len(criteria), "source": source, "criteria": criteria}
