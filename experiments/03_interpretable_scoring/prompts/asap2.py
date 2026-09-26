"""Elicitation and taxonomy prompts for the four ASAP 2.0 runs (paper Prompt ``prm:elicit-asap``).

Corpus twin of ``prompts/ellipse.py``. Structure, output schema, field order, parser and
taxonomy prompts are the same; only the corpus
framing, the source-article/assignment block, the out-of-scope list and the
"argumentative essay quality" wording differ. Everything else is held fixed so that
ELLIPSE-vs-ASAP is a CORPUS contrast, not a prompt contrast.

Two families, both self-contained in this file so the prompt is reviewable as a paper
artefact and cannot drift with shared code:

1. COMPARATIVE -- the pairwise judge. Shows two ASAP-2.0 "Driverless cars" essays and asks
   for (a) which is the stronger argumentative essay overall, and (b) the qualities that
   essay has more of than the other. Evidence-first per quality, uncapped.
2. TAXONOMY    -- groups the judge's raw quality phrases into canonical criteria by
   MEANING. Identical to the ELLIPSE prompt except the one framing clause naming
   the corpus. -> gpt-5.4, reasoning_effort=high, sync (background mode + poll).

WHY THIS SHAPE (do not "improve" it without re-reading the model). The fitted model is the
holistic-verdict model of ``core/model/holistic/``::

    P(w_p = 1)           = sigma( sum_k Delta_{p,k} + beta )
    P(c_{p,k} = 1 | w_p) = sigma( (2 w_p - 1) Delta_{p,k} + gamma_k )

The citation channel is *conditioned on the overall verdict* and signed by it via
eta_p = 2 w_p - 1. Two consequences for the elicitation:

  * There is no per-quality winner in the model, so none is elicited. Every cited quality
    favours the overall winner by construction -- that IS the generative story. A prompt
    eliciting a per-dimension winner would not fit: this likelihood has no parameter for it.
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

CORPUS CONTENT. The source article and the assignment are inlined verbatim below, as they
appear in the ASAP 2.0 release's ``source_text_1`` / ``assignment`` fields for the
"Driverless cars" prompt. They are inlined rather than rendered at judging time because
``scripts/01_judge.py`` loads this module by path and calls
``comparative_messages(essay_a, essay_b)`` with no context argument -- and because the
prompt must be frozen, byte-for-byte, as a paper artefact.
"""

from __future__ import annotations

import json
import re

# ---------------------------------------------------------------------------
# Corpus content (verbatim from the ASAP 2.0 release)
# ---------------------------------------------------------------------------

ASSIGNMENT = "In the article “Driverless Cars are Coming,” the author presents both positive and negative aspects of driverless cars. Using details from the article, create an argument for or against the development of these cars.  Be sure to include: your position on driverless cars; appropriate details from the article that support your position; an introduction, a body, and a conclusion to your argumentative essay."

SOURCE_ARTICLE = "Driverless Cars Are Coming \nCan you imagine a time in the future when no one buys cars because no one needs them anymore? Google cofounder Sergey Brin can. He envisions a future with a public transportation system where fleets of driverless cars form a public-transport taxi system. The cars he foresees would use half the fuel of today’s taxis and offer far more flexibility than a bus. He believes such cars would fundamentally change the world. \nTelevision and movies have long been fascinated with cars that could drive themselves. In reality, Google has had cars that could drive independently under specific conditions since 2009. Their cars have driven more than half a million miles without a crash, but so far, Google cars aren't truly driverless; they still alert the driver to take over when pulling in and out of driveways or dealing with complicated traffic issues, such as navigating through roadwork or accidents. So what roadblocks lie ahead for the autonomous car? \nSensing the World \nLet's begin by looking at which companies are making computer-driven cars. Originally, many futurists believed the key to developing self-driving cars someday wasn't so much smarter cars as smarter roads. For example, in the late 1950s, General Motors created a concept car that could run on a special test track. The track was embedded with an electrical cable that sent radio signals to a receiver on the front end of the car. Engineers at Berkeley tried something similar, but they used magnets with alternating polarity. The car read the positive and negative polarity as messages in binary code. These smart-road systems worked surprisingly well, but they required massive upgrades to existing roads, something that was simply too expensive to be practical. \nWithout the option of smarter roads, manufacturers turned to smarter cars—but how much smarter did the cars need to be? For starters, they needed a whole lot of sensors. Google's modified Toyota Prius uses position-estimating sensors on the left rear wheel, a rotating sensor on the roof, a video camera mounted near the rearview mirror, four automotive radar sensors, a GPS receiver, and an inertial motion sensor. The most important bit of technology in this system is the spinning sensor on the roof. Dubbed LIDAR, it uses laser beams to form a constantly updating 3-D model of the car's surroundings. The combination of all this input is necessary for the driverless car to mimic the skill of a human at the wheel. \nSensors are nothing new, of course. In the 1980s, automakers used speed sensors at the wheels in the creation of antilock brakes. Within 10 years, those sensors had become more advanced to detect and respond to the danger of out-of-control skids or rollovers. The information from the sensors can cause the car to apply brakes on individual wheels and reduce power from the engine, allowing far better response and control than a human driver could manage alone. Further improvements in sensors and computer hardware and software to make driving safer are also leading to cars that can handle more and more driving tasks on their own. \nDriving or Assisting? \nAntilock brakes and driver assistance still seem a long way from the dream of calling a driverless cab to take us wherever we desire, but Sebastian Thrun, founder of the Google Car project, believes that the technology has finally begun to catch up to the dream. “There was no way, before 2000, to make something interesting. The sensors weren't there, the computers weren't there, and the mapping wasn't there. Radar was a device on a hilltop that cost two hundred million dollars. It wasn't something you could buy at Radio Shack.” So just how driverless will the cars be in the near future?\nIn 2013, BMW announced the development of “Traffic Jam Assistant.” The car can handle driving functions at speeds up to 25 mph, but special touch sensors make sure the driver keeps hold of the wheel. In fact, none of the cars developed so far are completely driverless. They can steer, accelerate, and brake themselves, but all are designed to notify the driver when the road ahead requires human skills, such as navigating through work zones and around accidents. This means the human driver must remain alert and be ready to take over when the situation requires. This necessitates the car being ready to quickly get the driver's attention whenever a problem occurs. GM has developed driver's seats that vibrate when the vehicle is in danger of backing into an object. The Google car simply announces when the driver should be prepared to take over. Other options under consideration are flashing lights on the windshield and other heads-up displays. Manufacturers are also considering using cameras to watch that drivers are remaining focused on the road. While the driver watches the road, the car watches the driver. \nWhy would anyone want a driverless car that still needs a driver? Wouldn’t drivers get bored waiting for their turn to drive? “The psychological aspects of automation are really a challenge,” admits Dr. Werner Huber, a BMW project manager driver. “We have to interpret the driving fun in a new way.” Some manufacturers hope to do that by bringing in-car entertainment and information systems that use heads-up displays. Such displays can be turned off instantly when the driver needs to take over—something not available to drivers trying to text with a cell phone. In this way, the in-car system is actually a safety feature, and safety is a big concern. \nWaiting on the Law \nMost driving laws focus on keeping drivers, passengers, and pedestrians safe, and lawmakers know that safety is best achieved with alert drivers. Presently, traffic laws are written with the assumption that the only safe car has a human driver in control at all times. As a result, in most states it is illegal even to test computer-driven cars. California, Nevada, Florida, and the District of Columbia have led the country in allowing limited use of semi-autonomous cars; manufacturers believe that more states will follow as soon as the cars are proved more reliably safe. Still, even if traffic laws change, new laws will be needed in order to cover liability in the case of an accident. If the technology fails and someone is injured, who is at fault—the driver or the manufacturer? \nAutomakers are continuing their work on the assumption that the problems ahead will be solved. Tesla has projected a 2016 release for a car capable of driving on autopilot 90 percent of the time. Mercedes-Benz, Audi, and Nissan plan to have cars that can drive themselves by 2020. The road to the truly autonomous car stretches on ahead of us, but we grow closer to the destination every day."

# ---------------------------------------------------------------------------
# Shared fragments
# ---------------------------------------------------------------------------

# One construct per item: bundled items ("grammar and spelling") are unusable downstream
# because they cannot be assigned to a single canonical criterion. Identical to ELLIPSE.
_ANTI_BUNDLING = (
    "List each quality as exactly one item — do not combine qualities with "
    '"and", "/", "&", or a comma; split related qualities into separate items.'
)

# ASAP-2.0-specific exclusions. Three clauses are load-bearing:
#
#  * The side clause: every essay answers the SAME prompt and must take a for/against
#    position. Without this a judge scores agreement with the position, not the argument.
#    (ELLIPSE's analogous clause is about stance too, but ELLIPSE also needed a
#    writing-prompt clause because it spans 44 prompts; ASAP spans one, so that clause is
#    dropped -- there is no prompt difference left to exclude.)
#  * The detail-selection clause: with a shared source article, "which quotes were picked"
#    is a topic choice, not a quality -- except where it bears on argument quality, which
#    is exactly the carve-out kept here.
#  * The anonymization clause: ASAP-2.0/PERSUADE use PROPER_NAME / OTHER_PII /
#    STUDENT_NAME / LOCATION_NAME, NOT ELLIPSE's Generic_* convention. Verified against the
#    500-essay pool: they are RARE (0.2% of essays) versus ELLIPSE's 14.2%. Kept anyway,
#    symmetrically with the ELLIPSE prompt, so the two corpora differ only in which token
#    set is named -- not in whether the exclusion exists. It is held-constant boilerplate
#    here, not a corpus-motivated exclusion.
_OUT_OF_SCOPE = (
    "Out of scope — do not treat these as differences in quality:\n"
    "- the side an essay takes, for or against the development of driverless cars — "
    "either position can be argued well;\n"
    "- the topic itself, or which particular details from the article an essay chose to "
    "use, except insofar as those choices affect the quality of the argument;\n"
    "- length in itself;\n"
    "- anonymization placeholders inserted by the corpus builders to remove identifying "
    "information, such as PROPER_NAME, OTHER_PII, STUDENT_NAME, LOCATION_NAME or "
    "SCHOOL_NAME. Ignore them entirely: they are not the student's words. Never count them "
    "as spelling, capitalization, spacing or word-choice errors — including when a "
    "placeholder is misspelled or run together with an adjacent word."
)

_CORPUS_FRAMING = (
    "These are argumentative essays written by students in United States schools, "
    "primarily in grades 6, 8, 9 and 10. They span a wide range of writing ability and "
    "maturity, from beginning writers to quite proficient ones. Every essay responds to "
    "the same source-based argumentation task: the students read the article below and "
    "were asked to take and defend a position on it, using details from that article as "
    "evidence. Read and weigh the essays as you would when assessing argumentative "
    "writing at this level."
)

# The article and the assignment the students saw. Supplied verbatim so the judge can tell
# an accurate, well-integrated use of the source from a garbled or invented one.
_CONTEXT_BLOCK = (
    "── SOURCE ARTICLE ──\n" + SOURCE_ARTICLE + "\n"
    "\n"
    "── ASSIGNMENT GIVEN TO STUDENTS ──\n" + ASSIGNMENT
)


# ===========================================================================
# 1. COMPARATIVE — pairwise judge (verdict + the winner's qualities)
# ===========================================================================

COMPARATIVE_SYSTEM = (
    "You are helping build an interpretable scoring framework by examining how student "
    "argumentative essays differ in quality. The framework works by collecting many "
    "pairwise comparisons: for each pair, a judgment of which essay is stronger overall is "
    "recorded together with the qualities that essay has more of. Aggregated across many "
    "pairs, these reveal the distinct dimensions along which student writing varies. There "
    "is no fixed rubric — the qualities are recovered from comparisons like this one, not "
    "specified in advance.\n"
    "\n" + _CORPUS_FRAMING + "\n"
    "\n" + _CONTEXT_BLOCK + "\n"
    "\n"
    "Your task. You will be shown two student essays, Essay A and Essay B, both responding "
    "to this assignment. This pair is one comparison in a larger collection.\n"
    "\n"
    "First, decide which essay is the stronger argumentative essay overall. This is a "
    "forced choice: pick A or B even when the two are close.\n"
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
    "are comparable on some aspect of quality, leave it out rather than listing it. Do not "
    "pad the list to justify your choice, and do not list a quality on which the other "
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
    "Following the standard above, first give the essay that is the stronger argumentative "
    "essay overall, then list the qualities that essay has more of than the other — for "
    "each, first the concrete feature you saw, then the quality and a one-sentence "
    "description. Respond with the JSON object only."
)

# Property order == generation order: verdict first, then evidence before the label.
# Structurally IDENTICAL to the ELLIPSE schema; only `name` differs.
COMPARATIVE_SCHEMA = {
    "name": "argumentative_essay_comparison",
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
# IDENTICAL to the ELLIPSE taxonomy prompt except the one clause naming the corpus
# ("student argumentative essays, all written in response to one source-based prompt").
# The synonym example is kept verbatim so this prompt differs from ELLIPSE's by exactly
# that clause.
#
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

These qualities were produced by a judge comparing student argumentative essays, all written in
response to one source-based prompt. On each comparison the judge chose the stronger essay and
named the qualities that essay had more of. Aggregated across many comparisons, the same
underlying construct was named in many slightly different ways ("vocabulary", "lexical range",
"word choice precision" may all denote one construct). Your job is to group the different
phrasings of the SAME underlying construct together.

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
