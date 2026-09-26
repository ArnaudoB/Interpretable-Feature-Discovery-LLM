"""Prompts for the CMV comparative feature-discovery run (Prompts `prm:cmv-comp` and `prm:cmv-taxo`).

1. COMPARATIVE -- the pairwise judge. Neutral, evidence-first, per-dimension isolation, NO
   overall winner. Framed around *what drives persuasion*: it invites dimensions from BOTH the
   challenger's argument and the OP's own post. The setup is described generically (no platform
   name, no illustrative examples) to avoid priming the judge toward a known dataset or
   particular constructs. -> gpt-5.4-mini, batch.
2. TAXONOMY -- groups the judge's raw dimension phrases into canonical criteria by MEANING, with
   an "Other" catch-all and no illustrative examples. -> gpt-5.4 high, sync.

Response schemas are strict Responses-API object wrappers (``additionalProperties: false``,
every property required).

Each item's ``text`` is a ``[OP VIEW] … [CHALLENGE] …`` blob (see ``core.cmv.items``); the two
item texts fill the ``{arg_a}`` / ``{arg_b}`` slots below.
"""

from __future__ import annotations

# Shared: one dimension per item, no bundling.
_ANTI_BUNDLING = (
    "List each dimension as exactly one item — do not combine dimensions with "
    '"and", "/", "&", or a comma; split related dimensions into separate items.'
)

# Shared out-of-scope. Topic/subject-matter is the key exclusion here (the two exchanges
# are about different topics BY DESIGN); agreement and length are the other confounds.
_OUT_OF_SCOPE = (
    "Out of scope — do not treat these as dimensions of difference:\n"
    "- the topic or subject matter either exchange is about, and how interesting, "
    "important, difficult, or agreeable either OP's view is: the two exchanges concern "
    "different topics by design, so topic or subject-matter difference is never a "
    "dimension here;\n"
    "- which side you personally agree with, or whether you think either OP's view is "
    "correct;\n"
    "- the username or any other identifier of either poster or challenger;\n"
    "- length in itself."
)


# ===========================================================================
# 1. COMPARATIVE (debias, evidence-first, no verdict) — pairwise judge
# ===========================================================================
COMPARATIVE_SYSTEM = (
    "You are helping build an interpretable framework for what drives persuasion in "
    "debate, by examining how two attempts to change someone's mind differ. The framework "
    "works by collecting many pairwise comparisons: for each pair, the ways the two "
    "exchanges differ are articulated in plain terms, and aggregated across many pairs "
    "these reveal the distinct dimensions along which persuasion-relevant material varies. "
    "There is no fixed rubric — the dimensions are recovered from comparisons like this "
    "one, not specified in advance.\n"
    "\n"
    'Background. Each item is an online discussion in which an original poster (the "OP") '
    "states a view they currently hold and invites others to change it; a challenger then "
    "replies with an argument aimed at changing that specific view. You will be shown two "
    "such exchanges, Exchange A and Exchange B, each presented as a block that begins with "
    "the OP's view (marked [OP VIEW]) followed by the challenger's reply (marked "
    "[CHALLENGE]). Crucially, the two exchanges concern different posters holding different "
    "views on different topics.\n"
    "\n"
    "Your task. Identify the dimensions on which the two exchanges meaningfully differ in "
    "ways that could bear on whether the OP's view gets changed. Consider the whole "
    "exchange: both the challenger's argument and the OP's post itself. Report the "
    "differences that genuinely stand out to you on reading this pair — bring up whatever "
    "you find salient, and do not work from a predetermined list. Favor real, noticeable "
    "differences over minor ones, but do not force the list to be short: report as many as "
    "actually distinguish these two exchanges.\n"
    "\n"
    "For each dimension, first name the concrete, specific feature you actually saw — a "
    "short phrase pointing to the specific evidence. Then, grounded in that evidence, name "
    "the dimension, describe in one sentence what it captures, and indicate which exchange "
    "shows more of it.\n"
    "\n"
    "Judge each dimension on its own terms, from its own evidence. Exchanges have uneven "
    "profiles: it is entirely normal for one to be stronger on some dimensions and the "
    "other stronger on others, so let the exchange that shows more of each dimension fall "
    "wherever the evidence for that dimension points.\n"
    "\n" + _OUT_OF_SCOPE + "\n"
    "\n"
    "How to report. " + _ANTI_BUNDLING + " For the evidence field, give a short phrase "
    "naming the concrete feature you saw. For the dimension field, give a concise general "
    "noun phrase naming the dimension (the kind of short label you might use as a rubric "
    "column). For the description field, give one sentence saying what that dimension "
    "captures. Indicate which exchange shows more of it.\n"
    "\n"
    "Output only JSON of the form:\n"
    '{"differences": [{"evidence": "<short phrase>", "dimension": '
    '"<concise noun phrase>", "description": "<one sentence>", '
    '"winner": "A" | "B"}, ...]}'
)

COMPARATIVE_USER = (
    "Exchange A:\n{arg_a}\n\n"
    "Exchange B:\n{arg_b}\n\n"
    "Following the standard above, list the dimensions on which these two exchanges differ "
    "in ways bearing on persuasion — whether in the challenger's argument or in the OP's "
    "post — for each giving the concrete feature you saw, the dimension, a one-sentence "
    "description, and which exchange shows more of it. Compare on persuasion-relevant "
    "qualities, not topic. Respond with the JSON object only."
)

COMPARATIVE_SCHEMA = {
    "name": "cmv_argument_differences",
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


def comparative_messages(arg_a: str, arg_b: str) -> tuple[str, str]:
    """Return (system, user) for one comparative pair."""
    return COMPARATIVE_SYSTEM, COMPARATIVE_USER.format(arg_a=arg_a, arg_b=arg_b)


# ===========================================================================
# 2. TAXONOMY — group raw dimension phrases into canonical criteria (gpt-5.4 high)
#    (no illustrative examples: the grouping must not be primed toward any constructs)
# ===========================================================================
TAXONOMY_PROMPT = """You are organizing a flat list of evaluation dimensions into a set of canonical criteria.

These dimensions were produced by judges comparing pairs of online exchanges — in each, an
original poster (the "OP") states a view and a challenger replies with an argument aimed at changing
it. On each comparison a judge named the dimensions on which two exchanges differed in ways bearing
on whether the OP's view gets changed; these span both the challenger's argument and the OP's own
post. Aggregated across many comparisons, the same underlying construct was named in many slightly
different ways. Your job is to group the different phrasings of the SAME underlying construct
together.

Below is the full list. Each entry has an INDEX in square brackets, a short NAME, a one-sentence
DESCRIPTION of what it captures, and CITATIONS (how many times it was named — for your awareness
only).

[[INDEXED_DIMENSION_LIST]]

Instructions:
- Group entries that denote the SAME underlying construct, judging by MEANING, not by surface
  wording. Different words for the same idea belong in one group; the same word used for different
  ideas belongs in different groups.
- A dimension describing the CHALLENGER's argument and a dimension describing the OP's POST are
  DIFFERENT constructs even when they share a word — group by the specific feature and whose it is,
  keeping OP-side and argument-side constructs in separate groups.
- Do NOT force everything into a small number of groups, but do NOT split one construct into its
  facets either. A dimension that does not clearly share a construct with any other — including a
  framing or appeal specific to one debate's subject matter that does not recur as a general
  persuasion feature — does NOT become its own criterion: place all such genuinely-apart,
  idiosyncratic dimensions together in a single catch-all group whose canonical name is exactly
  "Other". Do not create many single-member groups, and do not absorb a genuinely-apart dimension
  into a loosely-related group to tidy up — send it to "Other".
- Do NOT merge two constructs just because they often co-occur or because persuasive exchanges tend
  to show both. Group only by shared MEANING, never by how exchanges might score.
- Citations are context only. Do not merge a frequently-named dimension into a rarely-named one or
  vice versa on the basis of counts; counts do not determine grouping.

For each group, give:
- a canonical NAME (a concise noun phrase suitable as a rubric column; use exactly "Other" for the
  catch-all group),
- a one-sentence DEFINITION of the construct the group captures,
- the list of members, each as its INDEX and its NAME copied verbatim from that index.

Rules on membership (these will be checked programmatically):
- Every input index must appear in exactly one group, exactly once. Place all of them; drop none;
  duplicate none.
- Do not invent indices that are not in the input.
- Each member's "name" must be the exact input NAME at that index — this is a cross-check, so copy
  it from the bracketed entry, do not paraphrase.

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
