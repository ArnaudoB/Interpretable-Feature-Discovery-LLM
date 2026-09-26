"""Engineered ground truth: the 17 leaf dials, their canonical construct sentences
(embedded for semantic matching), display labels, and the answer-key -> dial-matrix
builder.

The paper evaluates recovery over the ``D = 16`` RELEVANT dials: prose fluency is a
deliberate leakage check, not a role-relevant dial, so it is excluded from the dial
set for SF/SM/SR/AF/LK (it remains an off-target "halo" channel).
"""

from __future__ import annotations

import pandas as pd

# Canonical construct sentence per dial (embedded, matched against elicited criteria).
DIAL_TEXT = {
    "experience": "Years of professional software engineering work experience",
    "education": "Level of formal education attained such as bachelor's, master's or doctorate",
    "school_prestige": "Prestige or selectivity of the applicant's university",
    "employer_prestige": "Prestige or recognizability of the applicant's current or previous employer",
    "n_languages": "Breadth of programming languages the applicant is proficient in",
    "n_launches": "Number of production services the applicant has shipped or launched",
    "fluency": "Writing fluency and prose mechanics; clarity and freedom from grammatical errors",
    "skill_dsa_depth": "Depth of data structures and algorithms knowledge",
    "skill_python": "Proficiency and hands-on experience with the Python programming language",
    "skill_ai_agents_experience": "Hands-on experience building applied AI and large language model features",
    "skill_security_assessments": "Experience performing security assessments, threat modeling and code review",
    "skill_vuln_triage_automation": "Building automation for vulnerability report triage and routing",
    "skill_agentic_tooling_design": "Designing agentic tooling and autonomous multi-step agent pipelines",
    "side_project": "Personal side projects built outside of work",
    "open_source": "Open-source software contributions or project maintenance",
    "security_cert": "Security certifications such as OSCP or CompTIA Security Plus",
    "ctf": "Capture-the-flag competitions and offensive-security engagement",
}

DIAL_LABEL = {
    "experience": "Experience (years)",
    "education": "Education level",
    "school_prestige": "School prestige",
    "employer_prestige": "Employer prestige",
    "n_languages": "Languages known",
    "n_launches": "Production launches",
    "fluency": "Prose fluency",
    "skill_dsa_depth": "DSA depth",
    "skill_python": "Python",
    "skill_ai_agents_experience": "AI / agents experience",
    "skill_security_assessments": "Security assessments",
    "skill_vuln_triage_automation": "Vuln-triage automation",
    "skill_agentic_tooling_design": "Agentic tooling design",
    "side_project": "Side projects",
    "open_source": "Open-source",
    "security_cert": "Security certification",
    "ctf": "Capture-the-flag",
}

# Off-target channels that are not role-relevant dials but can leak into scores (halo).
HALO_SOURCES = ["fluency", "school_prestige", "employer_prestige"]


def _leaf_dials(answer_key: pd.DataFrame) -> list[str]:
    skills = [c for c in answer_key.columns if c.startswith("skill_")]
    niche = [
        c.replace("niche_", "").replace("_lvl", "")
        for c in answer_key.columns
        if c.startswith("niche_") and c.endswith("_lvl")
    ]
    return (
        [
            "experience",
            "education",
            "school_prestige",
            "employer_prestige",
            "n_languages",
            "n_launches",
            "fluency",
        ]
        + skills
        + niche
    )


def build_ground_truth(answer_key: pd.DataFrame) -> pd.DataFrame:
    """Numeric per-letter dial matrix (17 leaf dials) from ``answer_key.parquet``.

    ``answer_key`` must be indexed by ``letter_id``. Prose fluency is stored as
    ``-mistakes`` so that, like every other dial, a higher value is "better".
    """
    ak = answer_key
    gt = pd.DataFrame(index=ak.index)
    gt["experience"] = ak["YEARS"]
    gt["education"] = ak["edu_lvl"]
    gt["school_prestige"] = ak["school_lvl"]
    gt["employer_prestige"] = ak["employer_lvl"]
    gt["n_languages"] = ak["N_LANGUAGES"]
    gt["n_launches"] = ak["N_LAUNCHES"]
    gt["fluency"] = -ak["mistakes_applied"].astype(float)
    for c in [c for c in ak.columns if c.startswith("skill_")]:
        gt[c] = ak[c]
    for c in [c for c in ak.columns if c.startswith("niche_") and c.endswith("_lvl")]:
        gt[c.replace("niche_", "").replace("_lvl", "")] = ak[c]
    return gt[_leaf_dials(ak)]


def relevant_dials(answer_key: pd.DataFrame) -> list[str]:
    """The D=16 role-relevant dials used for recovery metrics (fluency excluded)."""
    return [d for d in _leaf_dials(answer_key) if d != "fluency"]
