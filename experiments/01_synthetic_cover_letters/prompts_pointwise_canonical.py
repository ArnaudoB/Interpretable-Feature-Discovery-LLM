"""Pointwise scoring against OUR canonicalized criteria — ablates the pairwise/BT machinery.

The judge rates ONE letter on all K canonical criteria in a single call, 0-10 each. No
comparison, no pair graph, no Bradley-Terry: the score is read straight off the model's
rating. Run against both criterion sets (the 32 the taxonomy yields once the "Other"
catch-all is removed, and the 21 the kappa/rho gate keeps), the arm answers "does the
statistical model earn its keep, holding the criteria fixed?"

**This is deliberately a near-copy of ``prompts.COMPARATIVE_SYSTEM``.** The framing
sentence, the job posting, the "uneven profiles" line and the out-of-scope list are carried
over wherever they still apply, so the contrast between the arms is the elicitation *mode*
and not incidental prompt wording. It follows the CMV pointwise arm (Experiment 2), which
is the same ablation on that dataset. Three things necessarily differ:

* One letter instead of two, so the pairwise-only lines are gone ("you will be shown two
  cover letters...", "which letter shows more of it").
* A scale has to be defined. Pairwise never needs one — "which shows more" is
  self-anchoring, which is precisely the property this arm gives up. The scale is anchored
  on the population of applications rather than on the letter, because the scores must be
  comparable *across* letters scored in separate calls; that is the whole difficulty of
  pointwise scoring and the prompt should not paper over it.
* An explicit push against central tendency, mirroring the comparative prompt's insistence
  that the winner "fall wherever the evidence points". Without it, pointwise ratings pile
  up at 5-7 and the arm loses to compression rather than to anything substantive.

Scale is 0-10, matching the CMV experiment's naive pointwise cells.
No per-score anchor descriptors: those would require a rubric-writing step our method never
performs, and would make this a richer arm rather than the pointwise counterpart of the
comparative prompt. That is what ``prompts.POINTWISE_SCORING_TEMPLATE`` (the M1/M2 arms)
does instead.

Every criterion is scored with a short evidence phrase emitted BEFORE the number, so the
phrase conditions the rating and the output length is uniform across reps.
"""

from __future__ import annotations

import json

import prompts as _p

JOB_AD = _p.JOB_AD
ROLE_LINE_ONE = (
    "The letter below is an application for a software-engineering role. The full job "
    "posting it responds to is:"
)

SCALE_LO, SCALE_HI = 0, 10
CRITERIA_SENTINEL = "[[CRITERIA]]"
INSTRUCTION_SENTINEL = "[[INSTRUCTION_VARIANT]]"

# Four reasoning-path variants cycled across reps with permuted criterion order -> a
# self-consistency ensemble, the same device the M1/M2 arms use
# (``prompts.POINTWISE_INSTRUCTION_VARIANTS``). Rewritten here because those defer to
# per-score descriptors, which these criteria deliberately do not have: every variant below
# rates on the one fixed, population-anchored 0-10 scale. All four emit an evidence phrase
# before the score, so reps are token-comparable.
POINTWISE_INSTRUCTION_VARIANTS = [
    {
        "key": "direct",
        "instruction": (
            "Rate the letter on each listed dimension independently, judging each on its own."
        ),
    },
    {
        "key": "evidence_first",
        "instruction": (
            "For each listed dimension, first note what in the letter bears on that dimension, and "
            "only then commit to a rating. Treat each dimension independently."
        ),
    },
    {
        "key": "population_referenced",
        "instruction": (
            "For each listed dimension, place the letter against applications of this kind in "
            "general -- not against this letter's other dimensions -- before settling on a rating."
        ),
    },
    {
        "key": "independent",
        "instruction": (
            "Take the dimensions one at a time, in the order listed, considering only what the "
            "letter shows on that specific dimension. Do not form an overall impression of the "
            "letter and do not let the other dimensions influence a rating."
        ),
    },
]

POINTWISE_CANONICAL_SYSTEM = (
    "You are helping apply an interpretable scoring framework for job applicants' cover "
    "letters. The framework scores a letter along a fixed set of dimensions, listed below.\n"
    "\n" + ROLE_LINE_ONE + "\n"
    "── JOB POSTING ──\n" + JOB_AD + "\n"
    "── END JOB POSTING ──\n"
    "\n"
    "The dimensions:\n" + CRITERIA_SENTINEL + "\n"
    "\n"
    "Your task. " + INSTRUCTION_SENTINEL + "\n"
    "\n"
    "For EACH listed dimension, in order, give a "
    "short evidence phrase (the concrete feature you saw in the letter) and then rate the "
    f"letter on that dimension from {SCALE_LO} to {SCALE_HI}.\n"
    "\n"
    "What the numbers mean. Rate each dimension relative to applications for this kind of "
    f"role in general, not relative to this letter's other dimensions: {SCALE_LO} means the "
    "dimension is entirely absent, 5 means it is present to a typical degree, and "
    f"{SCALE_HI} means it is present about as strongly as such an application ever gets. The "
    "same rating must mean the same thing on every letter you score, because these scores "
    "are compared across letters you will not see together.\n"
    "\n"
    "Judge each dimension on its own terms, from its own evidence. Applications have uneven "
    "profiles, so it is entirely normal for a letter to rate high on some dimensions and low "
    "on others. Use the full range — do not cluster every dimension in the middle. A "
    "dimension you cannot fully assess from this letter should still receive your best "
    "estimate, not a default middle rating.\n"
    "\n" + _p._OUT_OF_SCOPE_SCORING + "\n"
    "\n"
    "Return one item per listed dimension, in the same order, naming the dimension exactly "
    "as given, each with a short evidence phrase and an integer rating.\n"
    "\n"
    "Output only JSON of the form:\n"
    '{"criteria": [{"name": "<dimension>", "evidence": "<short phrase>", '
    f'"score": <integer {SCALE_LO}-{SCALE_HI}>}}, ...]}}'
)

POINTWISE_CANONICAL_USER = (
    "── COVER LETTER ──\n{letter}\n── END COVER LETTER ──\n\n"
    "Following the standard above, go through each listed dimension in order: give the "
    "concrete evidence you saw and then its rating. Respond with the JSON object only."
)

POINTWISE_CANONICAL_SCHEMA = {
    "name": "cover_letter_criteria_pointwise",
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
                        "score": {"type": "integer", "minimum": SCALE_LO, "maximum": SCALE_HI},
                    },
                },
            }
        },
    },
}


def render_criteria(criteria) -> str:
    """``criteria`` = [{name, definition}] -> numbered 'name: definition' list.

    Matches the CMV pointwise arm's renderer, so the two pointwise ablations render
    their panels identically.
    """
    return "\n".join(
        f"{i + 1}. {c['name']}: {c.get('definition', c.get('description', ''))}"
        for i, c in enumerate(criteria)
    )


def pointwise_canonical_messages(letter: str, criteria, variant) -> tuple[str, str]:
    """Return (system, user); criteria are baked into the (cacheable) system prompt."""
    system = POINTWISE_CANONICAL_SYSTEM.replace(
        CRITERIA_SENTINEL, render_criteria(criteria)
    ).replace(INSTRUCTION_SENTINEL, variant["instruction"])
    return system, POINTWISE_CANONICAL_USER.format(letter=letter)


def parse_pointwise(text: str, feature_names, lo: int = SCALE_LO, hi: int = SCALE_HI) -> dict:
    """Parse the JSON -> {criterion_name: int score}, clamped to [lo, hi].

    Aligns by exact name, falling back to positional order when the returned count matches
    but the names do not — the same tolerance the CMV arm applies.
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
