"""Data-informed feature elicitation, the S feature set (Prompt `prm:cmv-sample-elicit`).

The model is asked to name the dimensions on which replies differ, having been shown **as many
real exchanges as fit in one call**. It is the counterpart of ``prompts/prior.py``, which asks
the same question having seen nothing: no exchanges, no outcomes, no corpus description. The
resulting set is then scored by the same pointwise prompt the other PW arms use
(``prompts/score_pointwise.py``), so the only thing separating ``S + PW`` from ``P + PW`` is
**whether the elicitor saw data**. The participation-ratio gap between the discovered and prior
sets would otherwise confound the comparative discovery machinery with the plain fact of having
looked at the corpus; this arm isolates the second.

The prompt keeps an evidence-first, open-count design: the framing sentence, "attend to the ways
they differ", no fixed count, an anti-bundling clause, and evidence phrase before name. It has
no per-score level descriptors: this arm elicits *features only*; adding a rubric would change
the scoring prompt too and the contrast would stop being one factor.

Four lines are load-bearing:

* **"their order carries no meaning"** is the blindness guarantee. It is only true because
  ``sample_elicit.build_pair_blocks`` swaps A/B per pair: ``items.parquet`` is sorted
  ``(pair_id, side desc)``, so the delta winner is row 1 of all 1,200 pairs and table order alone
  would be a perfect cue. The sentence and the shuffle stand or fall together.
* **"more or less of ... or something a reply either shows or does not"** keeps presence/absence
  dimensions admissible. A "gradable, not yes/no" filter would evict exactly those, and many
  discovered CMV criteria are presence/absence moves (*Analogical reasoning*, *Rhetorical
  questioning*). Requiring gradedness would hand this arm the prior set's shape by instruction.
* **"a property of a reply, not of the poster's view"** keeps the set challenger-side, as the
  discovered sets are by construction. OP-side dimensions are constant within a matched pair, so
  they contribute exactly zero to the paired LR -- but they still vary across pairs, so they would
  inflate the participation ratio in favour of this arm.
* **The anti-padding clause** is the one of ``prompts/prior_open.py``, whose open-count runs
  show that an open count is safe here (about 16 criteria per draw).

Deliberately absent, as in ``prompts/prior.py``:

* **The corpus is never named.**
* **Outcomes and base rates.** The elicitor is never told that deltas exist, let alone which reply
  earned one. The sample spans the held-out split, so a label reveal here would leak the
  evaluation set into feature construction.
* **Families of dimension to look for.** Naming them would pre-load roughly the discovered
  set's structure and the arm would measure the hint instead of the data.
"""

from __future__ import annotations

import json

#: Below this a parse is treated as a failed call rather than an honest short panel.
MIN_FEATURES = 5

SAMPLE_SYSTEM = "You design an interpretable scoring rubric and output only JSON."

#: Replaced with the rendered ``Exchange k:`` blocks (``sample_elicit.render_exchanges``).
EXCHANGES_SENTINEL = "[[EXCHANGES]]"

SAMPLE_USER = """You are helping build an interpretable scoring framework by examining how attempts to change someone's mind differ from one another. There is no fixed set of dimensions — they are recovered from what actually distinguishes these exchanges, not specified in advance.

In online debates, a person states a view they hold and invites others to change it. Others reply with arguments aimed at changing that view.

Below is a sample of such exchanges. Each begins with the poster's view (marked [OP VIEW]), followed by two separate replies to it, written independently by different people (marked [REPLY A] and [REPLY B]). The two replies answer the same view; they are not responses to each other, and their order carries no meaning.

── EXCHANGES ──
[[EXCHANGES]]
── END EXCHANGES ──

Reading across them, attend to the ways the replies differ from one another — the axes along which you can see one reply showing more, less, or something different from another. Identify the dimensions on which these replies meaningfully differ as attempts to change the poster's view. Report whatever genuinely stands out to you; do not work from a predetermined list, and do not force the list to be short — report as many dimensions as actually distinguish these replies. Do not pad the list either: add a dimension only when it names something none of the others already capture.

A dimension may be something a reply shows more or less of along a range, or something a reply either shows or does not.

Every dimension must be a property of a reply, not of the poster's view or of the poster: name only dimensions on which the two replies to the same view can differ.

Keep genuinely distinct dimensions separate even when they are related or tend to co-occur — do not merge two dimensions because strong replies tend to show both. List each dimension as exactly one item — do not combine dimensions with "and", "/", "&", or a comma; split related dimensions into separate items.

For each dimension, first name the concrete, specific feature you actually saw in the replies — a short phrase pointing to the evidence. Then, grounded in that evidence, name the dimension and define in one sentence what it measures. Write the definition so a rater can apply it to a new reply without having seen this sample: describe the characteristic itself, and never refer to specific exchanges or to the reply labels.

Out of scope — do not treat these as dimensions of difference:
- the topic or subject matter, and how interesting, important, difficult or agreeable the poster's view is;
- which side you personally agree with, or whether the poster's view is correct;
- usernames or identifiers;
- length in itself.

Output only JSON:
{"criteria": [{"evidence": "<short phrase>", "name": "<short label, 2-5 words>", "definition": "<one sentence>"}, ...]}"""

SAMPLE_SCHEMA = {
    "name": "sample_persuasion_criteria",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["criteria"],
        "properties": {
            "criteria": {
                "type": "array",
                # No minItems/maxItems: the count is what this arm measures, so fixing it would
                # measure the instruction instead (feature_prior_root_open/REPORT.md).
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    # evidence first so the concrete observation is generated before the abstraction
                    "required": ["evidence", "name", "definition"],
                    "properties": {
                        "evidence": {"type": "string"},
                        "name": {"type": "string"},
                        "definition": {"type": "string"},
                    },
                },
            }
        },
    },
}


def sample_messages(exchanges_block: str) -> tuple[str, str]:
    """Return ``(system, user)`` with the rendered exchanges spliced in."""
    return SAMPLE_SYSTEM, SAMPLE_USER.replace(EXCHANGES_SENTINEL, exchanges_block)


def parse_sample(text: str, min_features: int = MIN_FEATURES) -> list:
    """Parse the JSON into ``[{name, definition, evidence}]``.

    Raises on a panel below ``min_features``: at that size the call failed (truncation, a refusal)
    rather than the model having honestly found few dimensions.
    """
    t = text.strip()
    if t.startswith("```"):
        t = t[t.find("{") : t.rfind("}") + 1]
    items = json.loads(t)["criteria"]
    out = []
    for it in items:
        nm, df = str(it["name"]).strip(), str(it["definition"]).strip()
        if nm and df:
            out.append(
                {"name": nm, "definition": df, "evidence": str(it.get("evidence", "")).strip()}
            )
    if len(out) < min_features:
        raise ValueError(
            f"parsed only {len(out)} criteria (min {min_features}) — treat as a failed call"
        )
    dupes = {c["name"] for c in out}
    if len(dupes) != len(out):
        raise ValueError("duplicate criterion names in the panel")
    return out


def to_features_json(criteria: list, source: str) -> dict:
    """Wrap a parsed panel in the ``features.json`` envelope the scoring pipeline reads.

    ``evidence`` is dropped here on purpose: it points at exchanges the scorer never sees. It
    stays in the draws file as part of the elicitation record.
    """
    return {
        "n_features": len(criteria),
        "source": source,
        "criteria": [{"name": c["name"], "definition": c["definition"]} for c in criteria],
    }
