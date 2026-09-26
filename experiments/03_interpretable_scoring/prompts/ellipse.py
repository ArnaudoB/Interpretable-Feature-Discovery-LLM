"""Elicitation and taxonomy prompts for the four ELLIPSE runs (paper Prompt ``prm:elicit-ellipse``).

Two families, both self-contained in this file so the prompt is reviewable as a paper
artefact and cannot drift with shared code:

1. COMPARATIVE -- the pairwise judge. Shows two ELL essays and asks for (a) which essay
   demonstrates stronger English language proficiency overall, and (b) the qualities that
   essay has more of than the other. Evidence-first per quality, uncapped.
   Reasoning is off on every judge (see runs.yaml).
2. TAXONOMY    -- groups the judge's raw quality phrases into canonical criteria by
   MEANING, with an "Other" catch-all and no target count.
   -> gpt-5.4, reasoning_effort=high, sync (background mode + poll).

WHY THIS SHAPE (do not "improve" it without re-reading the model). The fitted model is the
holistic-verdict model of ``core/model/holistic/``::

    P(w_p = 1)           = sigma( sum_k Delta_{p,k} + beta )
    P(c_{p,k} = 1 | w_p) = sigma( (2 w_p - 1) Delta_{p,k} + gamma_k )

The citation channel is *conditioned on the overall verdict* and signed by it via
eta_p = 2 w_p - 1. Two consequences for the elicitation:

  * There is no per-quality winner in the model, so none is elicited. Every cited quality
    favours the overall winner by construction -- that IS the generative story.
  * ``overall_winner`` is emitted FIRST, before the qualities. Property order in a strict
    Responses-API schema is generation order, so this makes the judge commit to the verdict
    and then enumerate what the winner has more of -- exactly the order the likelihood
    factorises in. Listing qualities first would make "the winner" undefined at the moment
    each quality is named.

The ``evidence`` field is emitted before ``quality`` for the same reason: it forces a
concrete observation before the abstract label, which reduces halo effects. The ``description``
field exists to give the taxonomy real semantic material to group on -- this pipeline does
NOT cluster in embedding space, so the description is the only signal beyond the bare name.

Response schemas are strict Responses-API object wrappers (``additionalProperties: false``,
every property required).
"""

from __future__ import annotations

import json
import re

# ---------------------------------------------------------------------------
# Shared fragments
# ---------------------------------------------------------------------------

# One construct per item: bundled items ("grammar and spelling") are unusable downstream
# because they cannot be assigned to a single canonical criterion.
_ANTI_BUNDLING = (
    "List each quality as exactly one item — do not combine qualities with "
    '"and", "/", "&", or a comma; split related qualities into separate items.'
)

# ELLIPSE-specific exclusions. Two clauses here are load-bearing:
#
#  * The writing-prompt clause: the corpus spans 44 distinct writing prompts and paired
#    essays routinely answer different ones, so without this the judge scores the task
#    instead of the writer.
#  * The anonymization clause: 14.2% of ELLIPSE essays (554/3911) carry placeholders, and
#    they often appear malformed or fused to a neighbouring word ("Generic_Namehad",
#    "wasGeneric_Name", "PROEPR_NAME"). Left unmentioned, a judge reads these as spelling,
#    capitalization or spacing errors -- contaminating precisely the criteria we most need
#    clean. NOTE: ELLIPSE does NOT use the "OTHER_PII" token (0 occurrences); that is an
#    ASAP-2.0/PERSUADE convention.
_OUT_OF_SCOPE = (
    "Out of scope — do not treat these as differences in proficiency:\n"
    "- the writing prompt an essay was responding to, or the topic or difficulty of that "
    "prompt — judge proficiency, not the task an essay happened to be assigned;\n"
    "- the position or opinion an essay takes — any stance can be expressed proficiently;\n"
    "- length in itself;\n"
    "- anonymization placeholders inserted by the corpus builders to remove identifying "
    "information, such as Generic_Name, Generic_City, Generic_School, PROPER_NAME, "
    "LOCATION_NAME, STUDENT_NAME, TEACHER_NAME or SCHOOL_NAME. Ignore them entirely: they "
    "are not the student's words. Never count them as spelling, capitalization, spacing or "
    "word-choice errors — including when a placeholder is misspelled or run together with "
    'an adjacent word (for example "Generic_Namehad" or "wasGeneric_Name").'
)

_CORPUS_FRAMING = (
    "These are essays written by English language learners in grades 8 to 12, during "
    "standardized writing assessments. They span a wide range of proficiency, from "
    "beginning to quite advanced English writers. Each essay responds to an independent "
    "writing prompt that requires no specialized background knowledge; different essays "
    "may respond to different prompts. Read and weigh the essays as you would when "
    "assessing the English writing proficiency of language learners at this level."
)


# ===========================================================================
# 1. COMPARATIVE — pairwise judge (verdict + the winner's qualities)
# ===========================================================================

COMPARATIVE_SYSTEM = (
    "You are helping build an interpretable scoring framework by examining how student "
    "essays differ in English language proficiency. The framework works by collecting many "
    "pairwise comparisons: for each pair, a judgment of which essay is stronger overall is "
    "recorded together with the qualities that essay has more of. Aggregated across many "
    "pairs, these reveal the distinct dimensions along which student writing varies. There "
    "is no fixed rubric — the qualities are recovered from comparisons like this one, not "
    "specified in advance.\n"
    "\n" + _CORPUS_FRAMING + "\n"
    "\n"
    "Your task. You will be shown two student essays, Essay A and Essay B. They may "
    "respond to different writing prompts. This pair is one comparison in a larger "
    "collection.\n"
    "\n"
    "First, decide which essay demonstrates stronger English language proficiency overall. "
    "This is a forced choice: pick A or B even when the two are close.\n"
    "\n"
    "Then identify the qualities that the essay you chose has more of than the other essay. "
    "Report the ones that genuinely stand out to you on reading this pair — bring up "
    "whatever you find salient, and do not work from a predetermined list. Favor real, "
    "noticeable differences over minor ones, but do not force the list to be short: report "
    "as many as actually distinguish these two essays.\n"
    "\n"
    "For each quality, first name the concrete, specific feature you actually saw in the "
    "essays — a short phrase pointing to the specific evidence. Then, grounded in that "
    "evidence, name the quality and describe in one sentence what it captures.\n"
    "\n"
    "Name only qualities on which the essay you chose is genuinely ahead. If the two essays "
    "are comparable on some aspect of proficiency, leave it out rather than listing it. Do "
    "not pad the list to justify your choice, and do not list a quality on which the other "
    "essay is in fact stronger.\n"
    "\n" + _OUT_OF_SCOPE + "\n"
    "\n"
    "How to report. " + _ANTI_BUNDLING + " For the evidence field, give a short phrase "
    "naming the concrete feature you saw. For the quality field, give a concise general "
    "noun phrase naming the quality (the kind of short label you might use as a rubric "
    "column). For the description field, give one sentence saying what that quality "
    "captures.\n"
    "\n"
    "Output only JSON of the form:\n"
    '{"overall_winner": "A" | "B", "qualities": [{"evidence": "<short phrase>", '
    '"quality": "<concise noun phrase>", "description": "<one sentence>"}, ...]}'
)

COMPARATIVE_USER = (
    "Essay A:\n{essay_a}\n\n"
    "Essay B:\n{essay_b}\n\n"
    "Following the standard above, first give the essay that demonstrates stronger English "
    "language proficiency overall, then list the qualities that essay has more of than the "
    "other — for each, first the concrete feature you saw, then the quality and a "
    "one-sentence description. Respond with the JSON object only."
)

# Property order == generation order: verdict first, then evidence before the label.
COMPARATIVE_SCHEMA = {
    "name": "essay_proficiency_comparison",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["overall_winner", "qualities"],
        "properties": {
            "overall_winner": {"type": "string", "enum": ["A", "B"]},
            "qualities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["evidence", "quality", "description"],
                    "properties": {
                        "evidence": {"type": "string", "maxLength": 160},
                        "quality": {"type": "string"},
                        "description": {"type": "string"},
                    },
                },
            },
        },
    },
}


def comparative_messages(essay_a: str, essay_b: str) -> tuple[str, str]:
    """Return ``(system, user)`` for one pair. Slot A is the essay shown first."""
    return COMPARATIVE_SYSTEM, COMPARATIVE_USER.format(essay_a=essay_a, essay_b=essay_b)


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def parse_comparative(text: str) -> dict | None:
    """Parse a comparative response.

    Returns ``{"overall_winner": "A"|"B", "qualities": [{evidence, quality, description}]}``
    or ``None`` when no usable JSON object with a valid verdict can be recovered.

    A missing or invalid ``overall_winner`` is fatal: without ``w_p`` the pair carries no
    information for either channel of the asymmetric model. Individual malformed quality
    items are dropped rather than failing the pair.
    """
    if not text:
        return None
    t = _FENCE_RE.sub("", text).strip()
    i, j = t.find("{"), t.rfind("}")
    if i == -1 or j == -1 or j <= i:
        return None
    try:
        obj = json.loads(t[i : j + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None

    winner = obj.get("overall_winner")
    if winner not in ("A", "B"):
        return None

    out: list[dict] = []
    for item in obj.get("qualities") or []:
        if not isinstance(item, dict):
            continue
        name = item.get("quality")
        if not isinstance(name, str) or not name.strip():
            continue
        out.append(
            {
                "evidence": str(item.get("evidence") or "").strip(),
                "quality": name.strip(),
                "description": str(item.get("description") or "").strip(),
            }
        )
    return {"overall_winner": winner, "qualities": out}


# ===========================================================================
# 2. TAXONOMY — canonicalize raw quality phrases (NO embedding clustering)
# ===========================================================================
# Groups by MEANING, with a programmatic membership cross-check, and an "Other" catch-all
# so idiosyncratic one-off phrases land in a single bucket instead of becoming many
# unidentifiable singleton criteria. Deliberately gives NO target number of criteria -- the
# model decides how many constructs there are.
#
# NO worked examples anywhere in this prompt, by design. Naming concrete constructs in the
# instructions ("cohesion", "grammatical accuracy", "vocabulary range", ...) primes the
# grouping toward them -- and those are exactly the constructs whose recovery IS the result.
# Do not add examples.

TAXONOMY_PROMPT = """You are organizing a flat list of evaluation qualities into a set of canonical criteria.

These qualities were produced by a judge comparing student essays for English language
proficiency. On each comparison the judge chose the stronger essay and named the qualities that
essay had more of. Aggregated across many comparisons, the same underlying construct was named
in many slightly different ways ("vocabulary", "lexical range", "word choice precision" may all
denote one construct). Your job is to group the different phrasings of the SAME underlying
construct together.

Below is the full list. Each entry has an INDEX in square brackets, a short NAME, a one-sentence
DESCRIPTION of what it captures, and CITATIONS (how many times it was named — for your awareness
only).

[[INDEXED_DIMENSION_LIST]]

Instructions:
- Group entries that denote the SAME underlying evaluative construct, judging by MEANING, not
  by surface wording. Different words for the same idea belong in one group; the same word
  used for different ideas belongs in different groups.
- Keep genuinely DISTINCT constructs separate, even when related. Two constructs a writing
  expert would score on separate rubric lines should be separate groups.
- Do NOT force everything into a small number of groups, but do NOT split one construct into
  its facets either. Use as many groups as there are distinct constructs.
- A quality that does not clearly share a construct with any other does NOT become its own
  criterion: place all such genuinely-apart, idiosyncratic qualities together in a single
  catch-all group whose canonical name is exactly "Other". Do not create many single-member
  groups, and do not absorb a genuinely-apart quality into a loosely-related group to tidy
  up — send it to "Other".
- Do NOT merge two constructs just because they often co-occur or because strong essays tend to
  score well on both. Group only by shared MEANING, never by how essays might score.
- Citations are context only. Do not merge a frequently-named quality into a rarely-named one
  or vice versa on the basis of counts; counts do not determine grouping.

For each group, give:
- a canonical NAME (a concise noun phrase suitable as a rubric column; use exactly "Other" for
  the catch-all group),
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


TAXONOMY_REPAIR_PROMPT = """The following qualities were left unplaced. Assign each to one of the EXISTING criteria
below (by exact canonical name) if it shares that construct, or to "Other" if it is genuinely
apart. Create a new group only if it is a distinct construct shared by several of the unplaced
qualities. Group only by MEANING.

EXISTING CRITERIA:
[[CRITERIA]]

UNPLACED QUALITIES:
[[LEFTOVERS]]

Output only JSON: {"criteria": [{"name": "<existing or new canonical name>", "definition": "<one sentence>",
"members": [{"index": <int>, "name": "<verbatim>"}, ...]}, ...]}  — include only the unplaced indices."""


TAXONOMY_SYSTEM = (
    "You are a careful taxonomist of writing-assessment constructs. "
    "Output only valid JSON matching the requested schema."
)


def render_dimension_list(entries) -> str:
    """Format ``(index, name, citations, description)`` rows for the taxonomy prompt."""
    return "\n".join(f"[{i}] {name} (cited {cites}x): {desc}" for i, name, cites, desc in entries)


def taxonomy_prompt(entries) -> str:
    return TAXONOMY_PROMPT.replace("[[INDEXED_DIMENSION_LIST]]", render_dimension_list(entries))


def taxonomy_repair_prompt(criteria, leftovers) -> str:
    crit_block = "\n".join(f"- {c['name']}: {c.get('definition', '')}" for c in criteria)
    return TAXONOMY_REPAIR_PROMPT.replace("[[CRITERIA]]", crit_block).replace(
        "[[LEFTOVERS]]", render_dimension_list(leftovers)
    )
