"""Contamination probe prompts (Table `tab:cmv-contamination`; Prompts `prm:cmv-contam-task` and
`prm:cmv-contam-named`).

The de-identification applied to both framings, the two system prompts (P1 unnamed, P2 naming
r/ChangeMyView and the delta mechanic), the shared user template and the answer schema, exactly
as the probe sent them. The probe's responses are shipped in
``raw/cells/contamination_probe_800/cache``; step 12 reads them.
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------- de-identification
# Remove tokens that NAME the platform / award mechanic (the giveaways P2 restores). The ordinary
# English phrase "change my/your view/mind" is left intact -- it describes the genre (P1's own prompt
# says the poster wants their mind changed) rather than naming the source; the P2-P1 lift, which holds
# text constant, is the clean signal regardless.
# The auto-appended moderator footer (present on ~60% of OPs) names the subreddit outright — remove the
# whole block first, then strip URLs (they contain reddit.com/r/changemyview), then residual tokens.
_FOOTER = re.compile(r"(?is)_{3,}.*?Happy CMVing!?\*?")
_FOOTER2 = re.compile(
    r"(?is)(Hello,?\s*users of|This is a footnote from the moderators).*?Happy CMVing!?\*?"
)
_MDLINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")  # [text](url) -> text
_URL = re.compile(r"https?://\S+")
_SCRUB = [
    (re.compile(r"(?i)changemyview"), " this forum "),
    (re.compile(r"(?i)cmv\w*:?"), " "),  # CMV, CMV:, CMVing, CMVs
    (re.compile(r"(?i)/?r/\s*\w+"), " a forum "),  # any /r/subreddit reference
    (re.compile(r"(?i)subreddit\w*"), " forum "),
    (re.compile(r"(?i)\breddit(ors?)?\b"), " "),
    (re.compile(r"(?i)\bdelta\s*bot\b"), " "),
    (re.compile(r"(?i)!?\bdeltas?\b"), " point "),
    (re.compile(r"[∆Δ]"), " "),
    (re.compile(r"(?i)\baward(ed|ing|s)?\b\s*(a\s*)?(point|view|change)?"), " "),
]


def deidentify(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = _FOOTER.sub(" ", text)
    text = _FOOTER2.sub(" ", text)
    text = _MDLINK.sub(r"\1", text)
    text = _URL.sub(" ", text)
    for pat, repl in _SCRUB:
        text = pat.sub(repl, text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


# --------------------------------------------------------------------------- prompts / schema
_TASK = (
    "You are shown one online discussion. It begins with a person's stated VIEW (a position they hold "
    "and are willing to reconsider), followed by ONE reply from someone arguing against that view. "
    "Your job: decide whether that reply actually changed the original person's view. Answer strictly "
    "yes or no."
)
P1_SYSTEM = _TASK
P2_SYSTEM = (
    "This is a post from the r/ChangeMyView (CMV) forum on Reddit. There, an original poster states a "
    'view and awards a "delta" (∆) to any reply that succeeds in changing their view. ' + _TASK
)
USER_TMPL = "[VIEW]\n{op}\n\n[REPLY]\n{reply}\n\nDid this reply change the original person's view? Answer yes or no."

SCHEMA = {
    "name": "view_changed",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["changed_view"],
        "properties": {"changed_view": {"type": "string", "enum": ["yes", "no"]}},
    },
}
