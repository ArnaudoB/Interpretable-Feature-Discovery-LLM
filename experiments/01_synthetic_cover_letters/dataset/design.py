"""Balanced, orthogonal experimental design over the corpus dials.

Produces one "profile" per letter: a level for each of the eight primary ordered
dials, a balanced-incomplete-block skill subset, a constrained niche assignment
(exactly ``present_niche`` of the niche dials non-absent), a nuisance template id,
and a nuisance signature name. The primary dials are made mutually near-orthogonal
by drawing several candidate assignments (seeded) and keeping the one with the
smallest max off-diagonal |correlation| over the non-structural column pairs --
a balanced assignment trick generalized to ordered
multi-level dials plus the constrained skill/niche blocks.

Two structural correlations are BY DESIGN and excluded from the objective/gate
(reported separately): (a) per-skill presence vs the skills-coverage-k dial (more
k => more skills), and (b) the niche presence indicators among themselves (exactly
2 of 4 present => sum is constant => pairwise corr -1/3). Everything else --
primary-vs-primary, primary-vs-skill, primary-vs-niche, skill-vs-niche -- is
minimized.

A small labeled set of EXTREME ANCHOR letters is appended after the balanced
design (anti-correlated "polished-but-thin" vs "rough-gem" profiles) to make the
halo legible; anchors are excluded from the balance counts and the orthogonality
gate.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from . import dials as D

NAMES = [
    "Jordan Reyes",
    "Alex Morgan",
    "Sam Delgado",
    "Taylor Brooks",
    "Casey Nguyen",
    "Riley Carter",
    "Morgan Ellis",
    "Jamie Foster",
    "Avery Sinclair",
    "Quinn Harper",
    "Drew Salazar",
    "Cameron Wells",
    "Reese Okafor",
    "Skyler Bennett",
    "Rowan Mercer",
    "Emerson Vance",
    "Hayden Cross",
    "Parker Lowe",
    "Sawyer Flynn",
    "Devon Ashby",
    "Kendall Rhodes",
    "Logan Priya",
    "Marlowe Kato",
    "Elliot Sato",
    "Blair Mensah",
    "Frankie Ibarra",
    "Toni Larsen",
    "Sasha Romano",
    "Micah Odom",
    "Noel Trent",
]


def _balanced(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Length-n vector over 0..k-1 with counts as equal as possible, shuffled."""
    base = np.tile(np.arange(k), n // k)
    rem = n - base.size
    if rem:
        base = np.concatenate([base, rng.permutation(k)[:rem]])
    rng.shuffle(base)
    return base


def _assign_skills(
    skillsk_lvl: np.ndarray, skills_k: list, n_skills: int, rng: np.random.Generator
) -> list:
    """Greedy usage-balanced skill subsets, orthogonal-by-construction to k."""
    n = len(skillsk_lvl)
    present: list = [None] * n
    usage = np.zeros(n_skills)
    for i in rng.permutation(n):
        k = skills_k[int(skillsk_lvl[i])]
        jitter = rng.random(n_skills) * 1e-6
        chosen = np.argsort(usage + jitter)[:k]
        usage[chosen] += 1
        present[i] = sorted(int(c) for c in chosen)
    return present


def _assign_niche(n: int, n_niche: int, present_count: int, rng: np.random.Generator) -> np.ndarray:
    """(n, n_niche) level matrix: 0 absent, 1 mid, 2 good; exactly present_count>0."""
    combos = list(combinations(range(n_niche), present_count))
    combo_idx = _balanced(n, len(combos), rng)
    lvl_stream = _balanced(n * present_count, 2, rng) + 1  # 1=mid, 2=good
    niche = np.zeros((n, n_niche), dtype=int)
    p = 0
    for i in range(n):
        for d in combos[combo_idx[i]]:
            niche[i, d] = int(lvl_stream[p])
            p += 1
    return niche


def _structural_mask(cols: list, n_skill: int, n_niche: int) -> np.ndarray:
    """True where a column pair's correlation is BY DESIGN (excluded from objective)."""
    K = len(cols)
    mask = np.zeros((K, K), dtype=bool)
    idx = {c: i for i, c in enumerate(cols)}
    skill_cols = [idx[f"skill_{j}"] for j in range(n_skill)]
    niche_cols = [idx[f"niche_{j}"] for j in range(n_niche)]
    ks = idx["skillsk_lvl"]
    for a in skill_cols:
        mask[a, ks] = mask[ks, a] = True  # skill vs k
        for b in skill_cols:
            mask[a, b] = True  # skill vs skill (BIB)
    for a in niche_cols:
        for b in niche_cols:
            mask[a, b] = True  # niche vs niche (2-of-4)
    return mask


def _build_columns(prim: dict, skills_present: list, niche: np.ndarray, n_skill: int) -> tuple:
    """Assemble the integer design matrix + column names for the corr objective."""
    n = len(next(iter(prim.values())))
    cols = list(D.PRIMARY_LVL_COLS)
    mats = [prim[c] for c in D.PRIMARY_LVL_COLS]
    skill_bin = np.zeros((n, n_skill), dtype=int)
    for i, ss in enumerate(skills_present):
        for c in ss:
            skill_bin[i, c] = 1
    for j in range(n_skill):
        cols.append(f"skill_{j}")
        mats.append(skill_bin[:, j])
    for j in range(niche.shape[1]):
        cols.append(f"niche_{j}")
        mats.append(niche[:, j])
    return np.column_stack(mats).astype(float), cols


def _max_offdiag(mat: np.ndarray, mask: np.ndarray) -> float:
    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(mat, rowvar=False)
    K = corr.shape[0]
    eye = np.eye(K, dtype=bool)
    consider = ~eye & ~mask
    vals = np.abs(corr[consider])
    return float(np.nanmax(vals)) if vals.size else 0.0


def _one_candidate(reg: D.Registry, n: int, ss: np.random.SeedSequence):
    rng = np.random.default_rng(ss)
    prim = {c: _balanced(n, 3, rng) for c in D.PRIMARY_LVL_COLS}
    skills_present = _assign_skills(prim["skillsk_lvl"], reg.skills_k, len(reg.skills_pool), rng)
    niche = _assign_niche(n, len(reg.niche_names), reg.present_niche, rng)
    templates = _balanced(n, len(reg.template_ids), rng)
    mat, cols = _build_columns(prim, skills_present, niche, len(reg.skills_pool))
    mask = _structural_mask(cols, len(reg.skills_pool), len(reg.niche_names))
    score = _max_offdiag(mat, mask)
    return score, prim, skills_present, niche, templates


def _realize(reg: D.Registry, prim, skills_present, niche, templates, rng, is_anchor, anchor_type):
    """Turn level indices into per-letter realized profiles (names, values, units)."""
    n = len(templates)
    profiles = []
    for i in range(n):
        exp = int(prim["exp_lvl"][i])
        edu = int(prim["edu_lvl"][i])
        sch = int(prim["school_lvl"][i])
        emp = int(prim["employer_lvl"][i])
        nl = int(prim["nlang_lvl"][i])
        nla = int(prim["nlaunch_lvl"][i])
        kk = int(prim["skillsk_lvl"][i])
        flu = int(prim["fluency_lvl"][i])
        school_band = D.SCHOOL_BAND_ORDER[sch]
        emp_band = D.EMPLOYER_BAND_ORDER[emp]
        prof = {
            "template_id": reg.template_ids[int(templates[i])],
            "is_anchor": bool(is_anchor),
            "anchor_type": anchor_type,
            "exp_lvl": exp,
            "edu_lvl": edu,
            "school_lvl": sch,
            "employer_lvl": emp,
            "nlang_lvl": nl,
            "nlaunch_lvl": nla,
            "skillsk_lvl": kk,
            "fluency_lvl": flu,
            "YEARS": reg.exp_years[exp],
            "DEGREE": D.EDUCATION_ORDER[edu],
            "school_band": school_band,
            "SCHOOL": str(rng.choice(reg.schools[school_band])),
            "employer_band": emp_band,
            "EMPLOYER": str(rng.choice(reg.employers[emp_band])),
            "N_LANGUAGES": reg.n_languages[nl],
            "N_LAUNCHES": reg.n_launches[nla],
            "k": reg.skills_k[kk],
            "fluency_count": reg.fluency_counts[flu],
            "skills_present": [reg.skills_pool[c] for c in skills_present[i]],
            "niche": {name: int(niche[i, j]) for j, name in enumerate(reg.niche_names)},
            "name": str(rng.choice(NAMES)),
        }
        profiles.append(prof)
    return profiles


# --------------------------------------------------------------------------- #
# Extreme anchors
# --------------------------------------------------------------------------- #
def _anchor_profiles(reg: D.Registry, n_anchors: int, rng: np.random.Generator):
    """Alternating anti-correlated anchors: 'polished-thin' (A) vs 'rough-gem' (B)."""
    profiles = []
    niche_names = reg.niche_names
    for a in range(n_anchors):
        typ = "A_polished_thin" if a % 2 == 0 else "B_rough_gem"
        tid = reg.template_ids[a % len(reg.template_ids)]
        if typ == "A_polished_thin":
            # great prose + prestige + high counts, but weak substance
            lv = dict(
                exp_lvl=0,
                edu_lvl=0,
                school_lvl=2,
                employer_lvl=2,
                nlang_lvl=2,
                nlaunch_lvl=2,
                skillsk_lvl=0,
                fluency_lvl=0,
            )
            niche_lvl = 1  # present-2 at 'mid'
            skills = [0, 1]  # k=2: first two skills
        else:
            # poor prose + low prestige + low counts, but strong substance
            lv = dict(
                exp_lvl=2,
                edu_lvl=2,
                school_lvl=0,
                employer_lvl=0,
                nlang_lvl=0,
                nlaunch_lvl=0,
                skillsk_lvl=2,
                fluency_lvl=2,
            )
            niche_lvl = 2  # present-2 at 'good'
            skills = list(range(len(reg.skills_pool)))  # k=6: all skills
        niche = np.zeros((1, len(niche_names)), dtype=int)
        niche[0, 0] = niche_lvl  # side_project
        niche[0, 3] = niche_lvl  # ctf
        prim = {c: np.array([lv[c]]) for c in D.PRIMARY_LVL_COLS}
        profiles += _realize(
            reg,
            prim,
            [skills],
            niche,
            np.array([reg.template_ids.index(tid)]),
            rng,
            is_anchor=True,
            anchor_type=typ,
        )
    return profiles


def build_design(
    reg: D.Registry, n_letters: int, n_anchors: int, seed: int, n_candidates: int = 24
) -> tuple:
    """Return (profiles, meta). ``profiles`` has ``n_letters`` entries (balanced +
    anchors); ``meta`` reports the chosen candidate and its max non-structural corr.
    """
    n_bal = n_letters - n_anchors
    master = np.random.SeedSequence(seed)
    ss_cand, ss_real, ss_anchor = master.spawn(3)

    best = None
    for i, cand in enumerate(ss_cand.spawn(n_candidates)):
        res = _one_candidate(reg, n_bal, cand)
        if best is None or res[0] < best[0]:
            best = (res[0], i) + res[1:]
    score, cand_idx, prim, skills_present, niche, templates = best

    rng_real = np.random.default_rng(ss_real)
    balanced = _realize(
        reg, prim, skills_present, niche, templates, rng_real, is_anchor=False, anchor_type=""
    )
    anchors = _anchor_profiles(reg, n_anchors, np.random.default_rng(ss_anchor))

    profiles = balanced + anchors
    meta = {
        "n_letters": n_letters,
        "n_balanced": n_bal,
        "n_anchors": n_anchors,
        "seed": seed,
        "n_candidates": n_candidates,
        "chosen_candidate_idx": cand_idx,
        "max_nonstructural_abs_corr": score,
    }
    return profiles, meta
