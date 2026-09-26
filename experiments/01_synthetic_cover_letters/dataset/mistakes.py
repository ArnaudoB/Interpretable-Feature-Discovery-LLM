"""Deterministic, length-preserving fluency degradation (the halo-source dial).

``inject_mistakes(text, n, seed, protect, protect_phrases)`` introduces exactly
``n`` within-token errors (typos) chosen deterministically from ``seed``. Design
invariants:

* **Whitespace/structure is preserved.** Text is split into word tokens and the
  whitespace between them (newlines, paragraph breaks) is kept verbatim; only the
  characters INSIDE chosen word tokens are mutated. So token count AND paragraph
  layout are identical before/after -> word count and formatting cannot drift
  across fluency levels; the injected-mistake COUNT is the only thing that varies.
* **No digits are ever introduced** (operators act on letters), so the digit
  whitelist in ``templates.validate`` still holds.
* **Protected tokens are never touched**: proper nouns (names, schools,
  employers) via ``protect`` (a lowercased set), and any word in a protected
  phrase (the verbatim authorization sentence) via ``protect_phrases``. Dial
  content in ordinary prose IS eligible; skill/niche detection runs on the CLEAN
  letter, so degradation cannot corrupt the answer key.

Errors are spread evenly across eligible tokens. Returns ``(degraded, applied)``;
``applied < n`` only if there are too few eligible tokens (a gate then flags it).
"""

from __future__ import annotations

import re

import numpy as np

MIN_LEN = 4  # only degrade reasonably long words
_VOWELS = "aeiou"

# whole-token common-confusion swaps (still exactly one token)
_HOMOPHONE = {
    "their": "thier",
    "there": "theyre",
    "receive": "recieve",
    "definitely": "definately",
    "separate": "seperate",
    "occurred": "occured",
    "necessary": "neccessary",
    "beginning": "begining",
    "believe": "beleive",
    "achieve": "acheive",
    "environment": "enviroment",
    "consistent": "consistant",
    "experience": "experiance",
    "independent": "independant",
}


def _is_eligible(tok: str, protect: set) -> bool:
    core = tok.strip(".,;:!?()'\"")
    if len(core) < MIN_LEN or not core.isalpha():
        return False
    if core.lower() in protect:
        return False
    return True


def _perturb(core: str, op: int) -> str:
    """Apply one character-level error to a word core (returns a changed word)."""
    low = core.lower()
    if low in _HOMOPHONE:
        out = _HOMOPHONE[low]
        return out.capitalize() if core[0].isupper() else out
    chars = list(core)
    n = len(chars)
    op = op % 3
    if op == 0:  # transpose two interior chars
        i = 1 + (n - 3) // 2
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    elif op == 1:  # drop an interior vowel
        for i in range(1, n - 1):
            if chars[i].lower() in _VOWELS:
                del chars[i]
                break
        else:  # no interior vowel -> double a char
            i = n // 2
            chars.insert(i, chars[i])
    else:  # double an interior consonant
        for i in range(1, n - 1):
            if chars[i].lower() not in _VOWELS:
                chars.insert(i, chars[i])
                break
        else:
            chars[1], chars[2] = chars[2], chars[1]
    out = "".join(chars)
    return out if out != core else core[::-1]  # guarantee an actual change


def _covered(words: list, phrases) -> set:
    """Word-list indices spanned by any protected phrase (verbatim contiguous)."""
    covered: set = set()
    for phrase in phrases:
        pt = phrase.split()
        if not pt:
            continue
        for i in range(len(words) - len(pt) + 1):
            if words[i : i + len(pt)] == pt:
                covered.update(range(i, i + len(pt)))
    return covered


def _perturb_piece(piece: str, op: int) -> str:
    """Perturb one whitespace-free token, preserving leading/trailing punctuation."""
    lead = ""
    core = piece
    while core and core[0] in "(\"'":
        lead += core[0]
        core = core[1:]
    trail = ""
    while core and core[-1] in ".,;:!?)'\"":
        trail = core[-1] + trail
        core = core[:-1]
    if len(core) < MIN_LEN or not core.isalpha():
        return piece
    return lead + _perturb(core, op) + trail


def inject_mistakes(
    text: str, n: int, seed: int, protect: set, protect_phrases=()
) -> tuple[str, int]:
    if n <= 0:
        return text, 0
    # keep words AND the whitespace between them (newlines/paragraphs preserved)
    pieces = re.split(r"(\s+)", text)
    word_pos = [i for i, p in enumerate(pieces) if p and not p.isspace()]
    words = [pieces[i] for i in word_pos]
    covered = _covered(words, protect_phrases)
    elig = [wi for wi, w in enumerate(words) if wi not in covered and _is_eligible(w, protect)]
    if not elig:
        return text, 0
    rng = np.random.default_rng(seed)
    m = min(n, len(elig))
    if m == len(elig):
        chosen = list(elig)
    else:
        stride = len(elig) / m
        off = float(rng.random()) * stride
        chosen = sorted({elig[min(len(elig) - 1, int(off + j * stride))] for j in range(m)})
        k = 0
        while len(chosen) < m:  # top up if rounding collided
            if elig[k] not in chosen:
                chosen.append(elig[k])
            k += 1
        chosen = sorted(chosen)
    applied = 0
    for rank, wi in enumerate(chosen):
        pi = word_pos[wi]
        new = _perturb_piece(pieces[pi], rank)
        if new != pieces[pi]:
            pieces[pi] = new
            applied += 1
    return "".join(pieces), applied
