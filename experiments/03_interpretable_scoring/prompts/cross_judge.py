"""Frozen prompt of the cross-judge consolidation step (paper Prompt ``prm:cross-judge``).

One GPT-5.4 (high reasoning) call per corpus groups the four judges' reliable criteria into
shared constructs by MEANING. The prompt deliberately carries NO statistics (no gamma, no
sigma_k, no citation rates), so the grouping cannot be contaminated by how essays scored, and
NO worked examples, for the same reason the elicitation prompts ban them.

Only the corpus clause differs between the two corpora. The ELLIPSE value is the one the
ELLIPSE call was made with, so both calls re-render byte-for-byte; ``scripts/
07_cross_judge_cluster.py`` (default replay mode) asserts it against the saved request.

Rendering and validation are generic and live in :mod:`core.analysis.cross_judge`.
"""

MODEL = "gpt-5.4-2026-03-05"
EFFORT = "high"
MAX_OUT = 40000

CLUSTER_SYSTEM = "You are a careful taxonomist of writing-assessment constructs. Output only valid JSON matching the requested schema."

CORPUS_FRAMING = {
    "ellipse": "the same student essays for English language proficiency",
    "asap2": "the same student argumentative essays, all written in response to one source-based prompt",
}

CLUSTER_PROMPT = """You are consolidating evaluation criteria discovered independently by several LLM judges into a set of shared constructs.

Each judge compared [[CORPUS_FRAMING]] and, with no fixed
rubric, named the qualities that distinguished the stronger essay. Each judge's raw phrasings
have already been canonicalized separately, so within a single judge the criteria below are
already distinct from one another. Your job is to identify, ACROSS judges, which criteria denote
the same underlying construct.

Below is the full list. Each entry has an INDEX in square brackets, the JUDGE that produced it,
its NAME, and a one-sentence DEFINITION.

[[INDEXED_CRITERION_LIST]]

Instructions:
- Group entries that denote the SAME underlying evaluative construct, judging by MEANING, not by
  surface wording. Different words for the same idea belong in one group; the same word used for
  different ideas belongs in different groups.
- Keep genuinely DISTINCT constructs separate, even when related. Two constructs a writing expert
  would score on separate rubric lines should be separate groups.
- A group may contain AT MOST ONE criterion per judge. The criteria within a judge were already
  canonicalized, so two of them never denote the same construct; if you find yourself wanting to
  place two criteria from one judge together, they belong in different groups.
- A criterion with no counterpart in any other judge forms a group of ONE. Do not absorb it into a
  loosely related group to tidy up, and do not create a catch-all: a construct only one judge
  discovered is a real finding and must stand alone.
- Do NOT try to balance the groups. There is no target number of groups and no expectation that
  every group contains every judge. Groups of one, two, three, and four are all equally valid
  outcomes; forcing coverage would fabricate agreement that is not there.
- Do NOT merge two constructs just because they often co-occur or because strong essays tend to
  score well on both. Group only by shared MEANING.
- The judge labels are present only so you can respect the one-per-judge rule. They carry no
  information about meaning; never group or split on the basis of which judge produced an entry.

For each group, give:
- a canonical NAME: a concise noun phrase naming the single construct the group captures. Name the
  construct itself, not the union of its members' wordings; avoid conjunctions unless the construct
  genuinely has no single name.
- a one-sentence DEFINITION of that construct.
- the list of members, each as its INDEX, its JUDGE, and its NAME copied verbatim.

Rules on membership (these will be checked programmatically):
- Every input index must appear in exactly one group, exactly once. Place all of them; drop none;
  duplicate none.
- Do not invent indices that are not in the input.
- Each member's "judge" and "name" must be the exact input values at that index -- this is a
  cross-check, so copy them from the bracketed entry, do not paraphrase.
- No group may contain two members with the same judge.

Output only JSON:
{"clusters": [{"name": "<canonical noun phrase>", "definition": "<one sentence>",
"members": [{"index": <int>, "judge": "<verbatim>", "name": "<verbatim>"}, ...]}, ...]}"""
