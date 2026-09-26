"""Ground-truth dial registry + CLEAN-text detectors for the cover-letter corpus.

Built at runtime from the (templates, job_ad) modules so this file carries no
hardcoded experiment paths. Every dial is an ORDERED factor: level index 0 is the
lowest tier. The registry maps level -> realized value (what goes into the letter)
and provides detectors that re-read the CLEAN (pre-degradation) letter to confirm
realized == assigned, catching any assembly/rendering bug (generation gets no
vote on truth).

Note on the fluency dial: its level index is a MISTAKE TIER (0=clean best prose,
1=mid, 2=degraded worst prose) -- higher level = WORSE, the reverse of every other
dial where higher = more/better. Analysis code must flip the sign accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass

# Ascending level order (index 0 = lowest tier) for the categorical dials.
SCHOOL_BAND_ORDER = ["less_selective", "moderately_selective", "highly_selective"]
EMPLOYER_BAND_ORDER = ["unknown_startup", "medium", "big_tech"]
EDUCATION_ORDER = ["BS", "MS", "PhD"]
# All three realized as TWO words so the education dial adds no length signal
# ("PhD" alone is one word -> would correlate education with letter length).
DEGREE_SLOT_TEXT = {"BS": "Bachelor's degree", "MS": "Master's degree", "PhD": "Doctoral degree"}
FLUENCY_ORDER = ["clean", "mid", "degraded"]  # mistake tier (0 = best prose)

# The eight primary ORDERED dials -- the scientifically-scored factors. Niche
# dials and per-skill presence are handled separately (constrained designs).
PRIMARY_DIALS = [
    "experience_years",
    "education",
    "school_band",
    "employer_band",
    "n_languages",
    "n_launches",
    "skills_coverage_k",
    "fluency",
]
# Column names for the primary dial LEVEL indices, in PRIMARY_DIALS order.
PRIMARY_LVL_COLS = [
    "exp_lvl",
    "edu_lvl",
    "school_lvl",
    "employer_lvl",
    "nlang_lvl",
    "nlaunch_lvl",
    "skillsk_lvl",
    "fluency_lvl",
]


@dataclass
class Registry:
    templates: object
    jobad: object
    exp_years: list  # e.g. [3, 5, 7]
    n_languages: list  # e.g. [2, 5, 9]
    n_launches: list  # e.g. [2, 4, 6]
    skills_k: list  # e.g. [2, 4, 6]
    fluency_counts: list  # [0, 5, 12] in FLUENCY_ORDER order
    skills_pool: list  # 6 skill keys
    niche_dials: dict  # name -> ["absent", mid, good]
    present_niche: int  # exactly this many niche dials non-absent per letter
    schools: dict  # band -> [names] (bands in ascending order)
    employers: dict  # band -> [names] (bands in ascending order)
    template_ids: list  # ["T1_classic", ...]

    @property
    def niche_names(self) -> list:
        return list(self.niche_dials.keys())


def build_registry(templates, jobad) -> Registry:
    fmc = templates.FLUENCY_MISTAKE_COUNTS
    return Registry(
        templates=templates,
        jobad=jobad,
        exp_years=list(jobad.EXPERIENCE_DIAL_YEARS),
        n_languages=list(jobad.N_LANGUAGES_DIAL),
        n_launches=list(jobad.N_LAUNCHES_DIAL),
        skills_k=list(jobad.SKILLS_MATCH_DIAL_K),
        fluency_counts=[fmc[k] for k in FLUENCY_ORDER],
        skills_pool=list(jobad.SKILLS_POOL),
        niche_dials={k: list(v) for k, v in templates.NICHE_DIALS.items()},
        present_niche=int(templates.PRESENT_NICHE_COUNT),
        schools={b: list(templates.SCHOOL_BANDS[b]) for b in SCHOOL_BAND_ORDER},
        employers={b: list(templates.EMPLOYER_BANDS[b]) for b in EMPLOYER_BAND_ORDER},
        template_ids=list(templates.TEMPLATES.keys()),
    )


# --------------------------------------------------------------------------- #
# CLEAN-text detectors: realized == assigned
# --------------------------------------------------------------------------- #
def _digits_multiset(text: str) -> list:
    """All standalone integer tokens in ``text`` (as ints)."""
    import re

    return sorted(int(m) for m in re.findall(r"(?<!\d)\d+(?!\d)", text))


def detect_mismatches(clean_text: str, prof: dict, reg: Registry) -> list:
    """Re-read a clean letter; return a list of realized!=assigned problems.

    Uses EXACT known realizations (unit sentences, names, degree text, digits) so
    there is zero detector cross-talk. Runs on the clean letter only; the degraded
    letter's integrity is covered by the length + mistake-count gates.
    """
    tpl = reg.templates.TEMPLATES[prof["template_id"]]
    problems: list[str] = []
    # unit sentences may embed slots (e.g. "At {EMPLOYER} I automated ..."); format
    # them the same way assemble() does before substring-matching against the letter.
    slots = {
        "NAME": prof["name"],
        "DEGREE": DEGREE_SLOT_TEXT[prof["DEGREE"]],
        "SCHOOL": prof["SCHOOL"],
        "YEARS": prof["YEARS"],
        "EMPLOYER": prof["EMPLOYER"],
        "N_LANGUAGES": prof["N_LANGUAGES"],
        "N_LAUNCHES": prof["N_LAUNCHES"],
    }

    # --- numbered dials: exactly the three whitelisted values, as a multiset ---
    expected_nums = sorted([prof["YEARS"], prof["N_LANGUAGES"], prof["N_LAUNCHES"]])
    if _digits_multiset(clean_text) != expected_nums:
        problems.append(
            f"digit multiset {_digits_multiset(clean_text)} != expected {expected_nums}"
        )

    # --- education: chosen degree text present, other two absent ---
    for code, txt in DEGREE_SLOT_TEXT.items():
        should = code == prof["DEGREE"]
        present = txt in clean_text
        if should != present:
            problems.append(f"education {code!r} present={present} expected={should}")

    # --- school / employer: chosen name present ---
    if prof["SCHOOL"] not in clean_text:
        problems.append(f"school {prof['SCHOOL']!r} not found")
    if prof["EMPLOYER"] not in clean_text:
        problems.append(f"employer {prof['EMPLOYER']!r} not found")

    # --- skills: each present skill's exact unit sentence present, absent absent ---
    for sk in reg.skills_pool:
        sent = tpl["skill_units"][sk].format(**slots)
        should = sk in prof["skills_present"]
        present = sent in clean_text
        if should != present:
            problems.append(f"skill {sk!r} present={present} expected={should}")

    # --- niche: present dials' exact unit at the assigned level; absent absent ---
    for dial, levels in reg.niche_dials.items():
        lvl = prof["niche"][dial]  # 0 absent, 1 mid, 2 good
        for li, lname in enumerate(levels):
            if lname == "absent":
                continue
            sent = tpl["niche_units"][f"{dial}.{lname}"].format(**slots)
            should = li == lvl
            present = sent in clean_text
            if should != present:
                problems.append(f"niche {dial}.{lname} present={present} expected={should}")

    return problems
