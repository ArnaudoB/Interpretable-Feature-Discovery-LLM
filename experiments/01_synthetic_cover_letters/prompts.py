"""Elicitation prompts for the cover-letter comparative-vs-pointwise study.

Three families, all grounded in the frozen job ad (``dataset.job_ad.JOB_AD``):

1. COMPARATIVE   -- the pairwise judge. Neutral, evidence-first, per-dimension isolation,
   NO overall winner. For each pair it names the dimensions on which the two letters
   differ and which letter shows more of each. -> gpt-5.4-mini, batch.
2. TAXONOMY      -- groups the judge's raw dimension phrases into canonical criteria by
   MEANING (with an "Other" catch-all so idiosyncratic dims do not each become a
   singleton criterion). -> gpt-5.4 high, sync.
3. POINTWISE_FEATURES / POINTWISE_SCORING -- the pointwise baseline. FEATURES elicits an
   evidence-first rubric of dimensions, each with a self-chosen 1..N descriptor scale;
   SCORING rates one letter per dimension on that scale, with four reasoning-path
   instruction variants cycled across reps (self-consistency ensemble, token-matched to
   the comparative arm). -> gpt-5.4 high (features) / gpt-5.4-mini (scoring), batch.

Response schemas are strict Responses-API object wrappers (``additionalProperties: false``,
every property required).
"""

from __future__ import annotations

from dataset.job_ad import JOB_AD

# Shared: one dimension per item, no bundling.
_ANTI_BUNDLING = (
    "List each dimension as exactly one item — do not combine dimensions with "
    '"and", "/", "&", or a comma; split related dimensions into separate items.'
)

# Shared out-of-scope. Deliberately does NOT list prose quality, school, or employer:
# whether the judge lets those bleed across criteria IS the phenomenon under study.
_OUT_OF_SCOPE = (
    "Out of scope — do not treat these as dimensions of difference:\n"
    "- the applicant's name or any personal identifier;\n"
    "- the closing sentence about work authorization / visa sponsorship, which is "
    "identical in every letter;\n"
    "- length in itself."
)

# Scoring-task phrasing of the same exclusions ("affect a score", not "a dimension of
# difference") — the scoring prompt rates against a fixed rubric, it does not name dimensions.
_OUT_OF_SCOPE_SCORING = (
    "Out of scope — do not let these affect any score:\n"
    "- the applicant's name or any personal identifier;\n"
    "- the closing sentence about work authorization / visa sponsorship, which is "
    "identical in every letter;\n"
    "- length in itself."
)

ROLE_LINE = (
    "The letters below are applications for the same software-engineering role. The full "
    "job posting they respond to is:"
)


# ===========================================================================
# 1. COMPARATIVE (debias, evidence-first, no verdict) — pairwise judge
# ===========================================================================
COMPARATIVE_SYSTEM = (
    "You are helping build an interpretable scoring framework by examining how job "
    "applicants' cover letters differ. The framework works by collecting many pairwise "
    "comparisons: for each pair, the ways the two letters differ are articulated in plain "
    "terms, and aggregated across many pairs these reveal the distinct dimensions along "
    "which the applications vary. There is no fixed rubric — the dimensions are recovered "
    "from comparisons like this one, not specified in advance.\n"
    "\n" + ROLE_LINE + "\n"
    "── JOB POSTING ──\n" + JOB_AD + "\n"
    "── END JOB POSTING ──\n"
    "\n"
    "Your task. You will be shown two cover letters, Letter A and Letter B, both written "
    "for the role above. This pair is one comparison in a larger collection.\n"
    "\n"
    "Identify the dimensions on which the two letters meaningfully differ as applications "
    "for this role. Report the differences that genuinely stand out to you on reading this "
    "pair — bring up whatever you find salient, and do not work from a predetermined list. "
    "Favor real, noticeable differences over minor ones, but do not force the list to be "
    "short: report as many as actually distinguish these two letters.\n"
    "\n"
    "For each dimension, first name the concrete, specific feature you actually saw in the "
    "letters — a short phrase pointing to the specific evidence. Then, grounded in that "
    "evidence, name the dimension, describe in one sentence what it captures, and indicate "
    "which letter shows more of it.\n"
    "\n"
    "Judge each dimension on its own terms, from its own evidence. Applications have uneven "
    "profiles: it is entirely normal for one letter to be stronger on some dimensions and "
    "the other letter stronger on others, so let the letter that shows more of each "
    "dimension fall wherever the evidence for that dimension points.\n"
    "\n" + _OUT_OF_SCOPE + "\n"
    "\n"
    "How to report. " + _ANTI_BUNDLING + " For the evidence field, give a short phrase "
    "naming the concrete feature you saw. For the dimension field, give a concise general "
    "noun phrase naming the dimension (the kind of short label you might use as a rubric "
    "column). For the description field, give one sentence saying what that dimension "
    "captures. Indicate which letter shows more of it.\n"
    "\n"
    "Output only JSON of the form:\n"
    '{"differences": [{"evidence": "<short phrase>", "dimension": '
    '"<concise noun phrase>", "description": "<one sentence>", '
    '"winner": "A" | "B"}, ...]}'
)

COMPARATIVE_USER = (
    "Letter A:\n{letter_a}\n\n"
    "Letter B:\n{letter_b}\n\n"
    "Following the standard above, list the dimensions on which these two cover letters "
    "differ as applications for the role — for each, first give the concrete feature you "
    "saw, then the dimension, a one-sentence description, and which letter shows more of "
    "it. Respond with the JSON object only."
)

COMPARATIVE_SCHEMA = {
    "name": "cover_letter_differences",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["differences"],
        "properties": {
            "differences": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["evidence", "dimension", "description", "winner"],
                    "properties": {
                        "evidence": {"type": "string"},
                        "dimension": {"type": "string"},
                        "description": {"type": "string"},
                        "winner": {"type": "string", "enum": ["A", "B"]},
                    },
                },
            }
        },
    },
}


# ===========================================================================
# 2. TAXONOMY — group raw dimension phrases into canonical criteria (gpt-5.4 high)
#    (no illustrative examples: the grouping must not be primed toward any constructs)
# ===========================================================================
TAXONOMY_PROMPT = """You are organizing a flat list of evaluation dimensions into a set of canonical criteria.

These dimensions were produced by judges comparing job applicants' cover letters for the
same software-engineering role. Each judge, on each comparison, named the dimensions on which
two letters differed. Aggregated across many comparisons, the same underlying construct was
named in many slightly different ways. Your job is to group the different phrasings of the
SAME underlying construct together.

Below is the full list. Each entry has an INDEX in square brackets, a short NAME, a one-sentence
DESCRIPTION of what it captures, and CITATIONS (how many times it was named — for your awareness
only).

[[INDEXED_DIMENSION_LIST]]

Instructions:
- Group entries that denote the SAME underlying evaluative construct, judging by MEANING, not
  by surface wording. Different words for the same idea belong in one group; the same word used
  for different ideas belongs in different groups.
- Do NOT force everything into a small number of groups, but do NOT split one construct into its
  facets either. A dimension that does not clearly share a construct with any other does NOT
  become its own criterion: place all such genuinely-apart, idiosyncratic dimensions together in a
  single catch-all group whose canonical name is exactly "Other". Do not create many single-member
  groups, and do not absorb a genuinely-apart dimension into a loosely-related group to tidy up —
  send it to "Other".
- Do NOT merge two constructs just because they often co-occur or because strong applicants tend
  to score well on both. Group only by shared MEANING, never by how applicants might score.
- Citations are context only. Do not merge a frequently-named dimension into a rarely-named one
  or vice versa on the basis of counts; counts do not determine grouping.

For each group, give:
- a canonical NAME (a concise noun phrase suitable as a rubric column; use exactly "Other" for the
  catch-all group),
- a one-sentence DEFINITION of the construct the group captures,
- the list of members, each as its INDEX and its NAME copied verbatim from that index.

Rules on membership (these will be checked programmatically):
- Every input index must appear in exactly one group, exactly once. Place all of them; drop
  none; duplicate none.
- Do not invent indices that are not in the input.
- Each member's "name" must be the exact input NAME at that index — this is a cross-check, so
  copy it from the bracketed entry, do not paraphrase.

Output only JSON:
{"criteria": [{"name": "<canonical noun phrase>", "definition": "<one sentence>",
"members": [{"index": <int>, "name": "<verbatim input name at that index>"}, ...]}, ...]}"""


TAXONOMY_REPAIR_PROMPT = """The following dimensions were left unplaced. Assign each to one of the EXISTING criteria
below (by exact canonical name) if it shares that construct, or put it in a new group if it is a
genuinely distinct construct. Group only by MEANING.

EXISTING CRITERIA:
[[CRITERIA]]

UNPLACED DIMENSIONS:
[[LEFTOVERS]]

Output only JSON: {"criteria": [{"name": "<existing or new canonical name>", "definition": "<one sentence>",
"members": [{"index": <int>, "name": "<verbatim>"}, ...]}, ...]}  — include only the unplaced indices."""


# ===========================================================================
# 2b. TAXONOMY (POOLING VARIANT) — fuse the features of E independent rubric elicitations into
#     canonical criteria, replacing the embedding-clustering fusion of 06_pool_rubrics.py.
#     Motivation: under agglomerative fusion the dials M3 fails to recover are lost to the
#     acceptance threshold, not to the elicitations (the E=27 union names 15.8/16 dials), so the
#     threshold is doing the damage. An LLM taxonomy fuses by MEANING and has no such knob.
#
#     Identical to TAXONOMY_PROMPT except for the provenance paragraph (independent elicitations,
#     not pairwise comparisons) and CITATIONS -> RUN (each feature comes from exactly one
#     elicitation, so there are no citation counts; the frequency caveat carries over in force,
#     plus the within-run distinctness prior). Every other instruction is byte-identical.
# ===========================================================================
TAXONOMY_POOL_PROMPT = """You are organizing a flat list of evaluation dimensions into a set of canonical criteria.

These dimensions were produced by independent rubric elicitations over job applicants' cover
letters for the same software-engineering role. Each elicitation read its own random sample of
letters and, working from scratch, named the dimensions on which those letters meaningfully
differed. Aggregated across many independent elicitations, the same underlying construct was
named in many slightly different ways. Your job is to group the different phrasings of the
SAME underlying construct together.

Below is the full list. Each entry has an INDEX in square brackets, a short NAME, a one-sentence
DESCRIPTION of what it captures, and RUN (which elicitation produced it — for your awareness
only).

[[INDEXED_DIMENSION_LIST]]

Instructions:
- Group entries that denote the SAME underlying evaluative construct, judging by MEANING, not
  by surface wording. Different words for the same idea belong in one group; the same word used
  for different ideas belongs in different groups.
- Do NOT force everything into a small number of groups, but do NOT split one construct into its
  facets either. A dimension that does not clearly share a construct with any other does NOT
  become its own criterion: place all such genuinely-apart, idiosyncratic dimensions together in a
  single catch-all group whose canonical name is exactly "Other". Do not create many single-member
  groups, and do not absorb a genuinely-apart dimension into a loosely-related group to tidy up —
  send it to "Other".
- Do NOT merge two constructs just because they often co-occur or because strong applicants tend
  to score well on both. Group only by shared MEANING, never by how applicants might score.
- Run labels are context only. Do not merge a dimension named by many runs into one named by few,
  or vice versa, on the basis of how often it recurs; recurrence does not determine grouping.
  Two entries from the SAME run were named as distinct dimensions by that run, so they are
  unlikely to be the same construct — but judge by MEANING, and merge them if they plainly are.

For each group, give:
- a canonical NAME (a concise noun phrase suitable as a rubric column; use exactly "Other" for the
  catch-all group),
- a one-sentence DEFINITION of the construct the group captures,
- the list of members, each as its INDEX and its NAME copied verbatim from that index.

Rules on membership (these will be checked programmatically):
- Every input index must appear in exactly one group, exactly once. Place all of them; drop
  none; duplicate none.
- Do not invent indices that are not in the input.
- Each member's "name" must be the exact input NAME at that index — this is a cross-check, so
  copy it from the bracketed entry, do not paraphrase.

Output only JSON:
{"criteria": [{"name": "<canonical noun phrase>", "definition": "<one sentence>",
"members": [{"index": <int>, "name": "<verbatim input name at that index>"}, ...]}, ...]}"""

OTHER_CRITERION = "Other"  # the catch-all group; carried but never scored (see 07_score_pointwise)


# ===========================================================================
# 3. POINTWISE FEATURES — evidence-first rubric elicitation. Recovers the dimensions on which
#    the sampled letters differ (uncapped), each with a self-chosen 1..N descriptor scale
#    grounded in the sample. gpt-5.4 high, sync.
# ===========================================================================
POINTWISE_FEATURES_SENTINEL = "[[LETTERS]]"

POINTWISE_FEATURES_PROMPT = (
    "You are helping build an interpretable scoring framework by examining how job applicants' "
    "cover letters differ. There is no fixed rubric — the dimensions are recovered from what "
    "actually distinguishes these letters, not specified in advance.\n"
    "── JOB POSTING ──\n" + JOB_AD + "\n"
    "── END JOB POSTING ──\n"
    "Below is a sample of cover letters submitted for this role, selected to span the full range "
    "of applicant quality.\n"
    "── COVER LETTERS ──\n" + POINTWISE_FEATURES_SENTINEL + "\n"
    "Reading across them, attend to the ways the letters differ from one another — the axes along "
    "which you can see one letter showing more, less, or something different from another. Identify "
    "the dimensions on which these letters meaningfully differ as applications for this role. Report "
    "whatever genuinely stands out to you; do not work from a predetermined list, and do not force "
    "the list to be short — report as many dimensions as actually distinguish these letters.\n"
    "A dimension may be something a letter shows more or less of along a range, or something a letter "
    "either shows or does not.\n"
    "Keep genuinely distinct dimensions separate even when they are related or tend to co-occur — "
    "do not merge two dimensions because strong letters tend to show both. " + _ANTI_BUNDLING + "\n"
    "For each dimension, first name the concrete, specific feature you actually saw in the letters "
    "— a short phrase pointing to the evidence. Then, grounded in that evidence, name the dimension "
    "and describe in one sentence what it captures. Then write a short assessment guideline a future "
    "reader will use to score a new letter on this dimension on a scale you need to specify: say what to look for, and describe "
    "what distinguishes each score of the scale you come up with. \n" + _OUT_OF_SCOPE + "\n"
    "Output only JSON:\n"
    '{"features": [{"evidence": "<short phrase>", "name": "<concise noun phrase>", '
    '"description": "<one sentence>", "levels": [{"score": 1, "descriptor": "<a letter at '
    'the weakest point of your scale>"}, {"score": 2, "descriptor": "<...>"}, ... up to the '
    "top score of the scale you chose]}, ...]}\n"
    "— for each dimension choose your own scale (score 1 = weakest, up to a top point N you decide) "
    "and give one descriptor per score point, from 1 to N."
)

POINTWISE_FEATURES_SCHEMA = {
    "name": "cover_letter_features",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["features"],
        "properties": {
            "features": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["evidence", "name", "description", "levels"],
                    "properties": {
                        "evidence": {"type": "string"},
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "levels": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["score", "descriptor"],
                                "properties": {
                                    "score": {"type": "integer"},
                                    "descriptor": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            }
        },
    },
}


# ===========================================================================
# 4. POINTWISE SCORING — score one letter against the guideline anchors, on each dimension's own
#    scale (1..N as the rubric lists; standard rubric = 1-5). (gpt-5.4-mini)
# ===========================================================================
ESSAY_SENTINEL = "[[LETTER]]"
INSTRUCTION_SENTINEL = "[[INSTRUCTION_VARIANT]]"
RUBRIC_SENTINEL = "[[RUBRIC_ORDERED]]"
OUTPUT_SPEC_SENTINEL = "[[OUTPUT_SPEC]]"

# Four reasoning-path variants cycled across reps with permuted rubric order -> a self-
# consistency ensemble token-matched to the comparative arm. Mirrors
# INSTRUCTION_VARIANTS_ELLIPSE_GUIDELINES. Each dimension is scored on ITS OWN scale: the rubric
# lists a descriptor for each point (1..N) of whatever scale that dimension defines, so the
# instructions never hardcode 1-5 — they defer to the descriptors listed for each dimension, so a
# dimension may carry any self-defined number of points N.
# Each carries a `mode` that selects the OUTPUT shape: 'note_score' emits a brief note
# BEFORE the score (so the evidence-based paths reason before committing a number); 'score' emits
# a bare score. The output spec AND the response schema follow the mode.
POINTWISE_INSTRUCTION_VARIANTS = [
    {
        "key": "direct",
        "mode": "score",
        "instruction": (
            "Score the letter on each rubric dimension below independently. Each dimension lists a "
            "descriptor for every point of its own score scale; assign the integer score whose "
            "descriptor best matches the letter on that dimension. Judge each dimension on its own."
        ),
    },
    {
        "key": "evidence_first",
        "mode": "note_score",
        "instruction": (
            "For each rubric dimension below, briefly note what in the letter bears on that "
            "dimension, then assign the integer score, from that dimension's own listed scale, whose "
            "descriptor best matches the letter on that dimension. Treat each dimension independently."
        ),
    },
    {
        "key": "standard_referenced",
        "mode": "note_score",
        "instruction": (
            "For each rubric dimension below, compare the letter against the score descriptors listed "
            "for that dimension, then assign the integer score whose descriptor it matches most "
            "closely. Judge each dimension on its own."
        ),
    },
    {
        "key": "independent",
        "mode": "score",
        "instruction": (
            "Go through the rubric dimensions one at a time, in the order listed. For each dimension, "
            "consider only what the letter shows on that specific dimension and assign the integer "
            "score, from that dimension's own listed scale, whose descriptor best matches; do not form "
            "an overall impression of the letter and do not let the other dimensions influence the "
            "score."
        ),
    },
]

POINTWISE_SCORING_TEMPLATE = (
    "You are scoring a cover letter against a fixed rubric, for the software-engineering "
    "role below.\n"
    "── JOB POSTING ──\n" + JOB_AD + "\n"
    "── END JOB POSTING ──\n"
    "── COVER LETTER ──\n" + ESSAY_SENTINEL + "\n" + INSTRUCTION_SENTINEL + "\n"
    "A dimension you cannot fully assess from this letter should still receive your best "
    "estimate, not a default middle score. Score each dimension independently, and do not "
    "let the score on one dimension affect the scores on others. Applications have uneven "
    "profiles: it is possible for a letter to score low on some dimensions and high on "
    "others.\n"
    "── RUBRIC DIMENSIONS (score in the order listed; each lists the descriptors for its own "
    "score scale) ──\n"
    + RUBRIC_SENTINEL
    + "\n"
    + _OUT_OF_SCOPE_SCORING
    + "\n"
    + OUTPUT_SPEC_SENTINEL
)


def _output_spec(mode: str) -> str:
    """Variant-dependent 'Output only JSON …' block (note-before-score vs bare score)."""
    head = "Output only JSON of the form:\n"
    if mode == "score":
        return (
            head + '{"scores": [{"dimension": "<rubric dimension name>", '
            '"score": <integer on that dimension\'s own scale>}, ...]}\n'
            "— one entry per rubric dimension, in the order listed."
        )
    if mode == "note_score":
        return (
            head + '{"scores": [{"dimension": "<rubric dimension name>", '
            '"note": "<brief note>", "score": <integer on that dimension\'s own scale>}, ...]}\n'
            "— one entry per rubric dimension, in the order listed; write the note "
            "before the score."
        )
    raise ValueError(f"unknown output mode {mode!r}")


def _scores_schema(name: str, item_props: dict, required: list) -> dict:
    return {
        "name": name,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["scores"],
            "properties": {
                "scores": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": required,
                        "properties": item_props,
                    },
                }
            },
        },
    }


# No fixed maximum: each dimension is scored on its own 1..N scale (N chosen per dimension at
# elicitation time).
_SCORE_INT = {"type": "integer", "minimum": 1}
POINTWISE_SCORING_SCHEMA_SCORE = _scores_schema(
    "cover_letter_scores",
    {"dimension": {"type": "string"}, "score": _SCORE_INT},
    ["dimension", "score"],
)
POINTWISE_SCORING_SCHEMA_NOTE = _scores_schema(
    "cover_letter_scores_noted",
    {"dimension": {"type": "string"}, "note": {"type": "string"}, "score": _SCORE_INT},
    ["dimension", "note", "score"],
)


def scoring_schema_for(variant: dict) -> dict:
    """The strict response schema matching a variant's output mode."""
    return (
        POINTWISE_SCORING_SCHEMA_NOTE
        if variant.get("mode") == "note_score"
        else POINTWISE_SCORING_SCHEMA_SCORE
    )


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #
def comparative_messages(letter_a: str, letter_b: str) -> tuple[str, str]:
    return COMPARATIVE_SYSTEM, COMPARATIVE_USER.format(letter_a=letter_a, letter_b=letter_b)


def pointwise_features_prompt(letters_block: str) -> str:
    """Evidence-first rubric elicitation; each dimension gets a self-chosen 1..N scale."""
    return POINTWISE_FEATURES_PROMPT.replace(POINTWISE_FEATURES_SENTINEL, letters_block)


def format_rubric(features: list[dict]) -> str:
    """Render frozen features (each with its 1-5 level descriptors) into the rubric block."""
    lines = []
    for f in features:
        lines.append(f"- {f['name']}: {f['description']}")
        for lv in sorted(f.get("levels") or [], key=lambda d: d.get("score", 0)):
            lines.append(f"    {lv['score']}: {lv['descriptor']}")
    return "\n".join(lines)


def pointwise_scoring_prompt(letter: str, features_or_block, variant: dict) -> str:
    rubric_block = (
        features_or_block
        if isinstance(features_or_block, str)
        else format_rubric(features_or_block)
    )
    return (
        POINTWISE_SCORING_TEMPLATE.replace(ESSAY_SENTINEL, letter)
        .replace(INSTRUCTION_SENTINEL, variant["instruction"])
        .replace(RUBRIC_SENTINEL, rubric_block)
        .replace(OUTPUT_SPEC_SENTINEL, _output_spec(variant["mode"]))
    )
