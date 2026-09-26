"""Zero-shot baseline prompt (the "Zero--shot" row of tab:cmv-predictive).

The judge (gpt-5.4-mini-2026-03-17, max 500 output tokens) sees the OP's view and both replies
of a held-out pair and names the one that earned the delta; each pair is queried in both
presentation orders and the two answers are averaged. The responses are shipped in
``raw/cells/zeroshot_pairwise``; step 9 reads their per-pair verdicts.
"""

from __future__ import annotations

MODEL = "gpt-5.4-mini-2026-03-17"
MAX_OUT = 500

SYSTEM = """You are assessing which of two counterarguments actually succeeded in changing someone's mind.

Background. You will be shown one online discussion: a person (the "poster") states a view they currently hold and invites others to change it. Two different people replied, each arguing against that view. Exactly one of the two replies led the poster to state that their view had changed; the other did not.

Your task. Read the poster's view and both replies, then decide which reply changed the poster's mind. Exactly one of them did, so you must choose A or B — do not decline, hedge, or call it a tie.

Judge what actually moved this particular poster, not which argument you personally find stronger and not which side you agree with. Both replies answer the same view, so the topic itself carries no information here.

Out of scope — do not let these decide your answer:
- the username or any other identifier of the poster or either replier;
- the length of a reply in itself.

Output only JSON of the form:
{"reasoning": "<one sentence>", "winner": "A" | "B"}"""

USER = """[POSTER'S VIEW]
{op}

[REPLY A]
{a}

[REPLY B]
{b}

Which reply changed the poster's view? Respond with the JSON object only."""

SCHEMA = {
    "name": "cmv_pair_verdict",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["reasoning", "winner"],
        "properties": {
            "reasoning": {"type": "string"},
            "winner": {"type": "string", "enum": ["A", "B"]},
        },
    },
}
