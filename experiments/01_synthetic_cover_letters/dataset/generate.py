"""Assemble the cover-letter corpus from profiles, degrade fluency, run the gates.

``generate_dataset(templates, job_ad, cfg)`` is the single entry point (the
templates/job_ad modules are injected so this file has no hardcoded experiment
paths). It returns ``(df, meta)`` where ``df`` has one row per letter with the
full answer key + clean and degraded text. ``run_gates(df, meta, reg, cfg)``
enforces the dataset gates and returns a verdict dict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import dials as D
from . import design as DZ
from .mistakes import inject_mistakes

# proper nouns / auth words never eligible for typo injection (lowercased)
_FIXED_PROTECT = {
    "google",
    "singapore",
    "authorized",
    "sponsorship",
    "require",
    "visa",
    "fully",
    "information",
}


def _core_keys(prof: dict, reg: D.Registry, tpl: dict, rng: np.random.Generator) -> list:
    """6 core unit keys = the k skill keys + (6-k) LENGTH-MATCHED fillers, shuffled.

    The (6-k) fillers are chosen to make the total core word count as close as
    possible to the fixed k=6 target (sum of all six skill units), so that neither
    the skills-coverage-k dial nor which specific skills are present leaks through
    letter length. k=6 hits the target exactly (no fillers); k=4 picks the best 2
    of 4 fillers; k=2 is forced to all 4 (its residual is the only length wiggle).
    """
    from itertools import combinations

    k = prof["k"]
    skill_units, fillers = tpl["skill_units"], tpl["filler_units"]
    target = sum(len(skill_units[s].split()) for s in reg.skills_pool)
    present_len = sum(len(skill_units[s].split()) for s in prof["skills_present"])
    need = 6 - k
    best = None
    for combo in combinations(range(len(fillers)), need):
        tot = present_len + sum(len(fillers[c].split()) for c in combo)
        d = abs(tot - target)
        if best is None or d < best[0]:
            best = (d, combo)
    keys = list(prof["skills_present"]) + [f"filler_{c}" for c in best[1]]
    assert len(keys) == 6, (k, keys)
    rng.shuffle(keys)
    return keys


def _niche_keys(prof: dict, reg: D.Registry, rng: np.random.Generator) -> list:
    keys = []
    for dial, lvl in prof["niche"].items():
        if lvl > 0:
            lname = reg.niche_dials[dial][lvl]
            keys.append(f"{dial}.{lname}")
    rng.shuffle(keys)
    return keys


def _protect_set(prof: dict) -> set:
    protect = set(_FIXED_PROTECT)
    for field in ("SCHOOL", "EMPLOYER", "name"):
        for w in str(prof[field]).replace(".", " ").split():
            protect.add(w.lower())
    return protect


def generate_dataset(templates, job_ad, cfg: dict) -> tuple:
    reg = D.build_registry(templates, job_ad)
    gen = cfg["generation"]
    n_letters = int(cfg["cell"]["n_letters"])
    seed = int(cfg["cell"]["seed"])
    n_anchors = int(gen["n_anchors"])
    n_candidates = int(gen.get("n_assignment_candidates", 24))

    profiles, meta = DZ.build_design(reg, n_letters, n_anchors, seed, n_candidates)

    # deterministic per-letter content/mistake seeds from the master seed
    ss_content, ss_mistake = np.random.SeedSequence(seed).spawn(2)
    content_seeds = np.random.default_rng(ss_content).integers(0, 2**31, size=len(profiles))
    mistake_seeds = np.random.default_rng(ss_mistake).integers(0, 2**31, size=len(profiles))

    rows = []
    for i, prof in enumerate(profiles):
        rng = np.random.default_rng(int(content_seeds[i]))
        slots = {
            "NAME": prof["name"],
            "DEGREE": D.DEGREE_SLOT_TEXT[prof["DEGREE"]],
            "SCHOOL": prof["SCHOOL"],
            "YEARS": prof["YEARS"],
            "EMPLOYER": prof["EMPLOYER"],
            "N_LANGUAGES": prof["N_LANGUAGES"],
            "N_LAUNCHES": prof["N_LAUNCHES"],
        }
        core = _core_keys(prof, reg, templates.TEMPLATES[prof["template_id"]], rng)
        niche = _niche_keys(prof, reg, rng)
        clean = templates.assemble(prof["template_id"], slots, core, niche)
        degraded, applied = inject_mistakes(
            clean,
            prof["fluency_count"],
            int(mistake_seeds[i]),
            _protect_set(prof),
            protect_phrases=[templates.AUTH_SENTENCE],
        )
        row = {
            "letter_id": f"L{i:04d}",
            "template_id": prof["template_id"],
            "is_anchor": prof["is_anchor"],
            "anchor_type": prof["anchor_type"],
            **{c: prof[c] for c in D.PRIMARY_LVL_COLS},
            "YEARS": prof["YEARS"],
            "DEGREE": prof["DEGREE"],
            "school_band": prof["school_band"],
            "SCHOOL": prof["SCHOOL"],
            "employer_band": prof["employer_band"],
            "EMPLOYER": prof["EMPLOYER"],
            "N_LANGUAGES": prof["N_LANGUAGES"],
            "N_LAUNCHES": prof["N_LAUNCHES"],
            "k": prof["k"],
            "fluency_count": prof["fluency_count"],
            "mistakes_applied": applied,
            "skills_present": ";".join(prof["skills_present"]),
            "name": prof["name"],
            "n_words": len(degraded.split()),
            "text_clean": clean,
            "text": degraded,
            "_profile": prof,  # kept for the realized==assigned gate; dropped before save
        }
        for sk in reg.skills_pool:
            row[f"skill_{sk}"] = int(sk in prof["skills_present"])
        for dial in reg.niche_names:
            row[f"niche_{dial}_lvl"] = prof["niche"][dial]
        rows.append(row)

    df = pd.DataFrame(rows)
    meta["registry"] = {
        "primary_dials": D.PRIMARY_DIALS,
        "skills_pool": reg.skills_pool,
        "niche_dials": reg.niche_names,
        "fluency_counts": reg.fluency_counts,
    }
    return df, meta


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #
def _orthogonality(df_bal: pd.DataFrame, reg_names: dict) -> tuple:
    cols = list(D.PRIMARY_LVL_COLS)
    cols += [f"skill_{s}" for s in reg_names["skills_pool"]]
    cols += [f"niche_{d}_lvl" for d in reg_names["niche_dials"]]
    mat = df_bal[cols].to_numpy(float)
    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(mat, rowvar=False)
    K = len(cols)
    # structural mask: skill-vs-skill, skill-vs-skillsk, niche-vs-niche
    mask = np.zeros((K, K), bool)
    ci = {c: i for i, c in enumerate(cols)}
    sk = [ci[f"skill_{s}"] for s in reg_names["skills_pool"]]
    ni = [ci[f"niche_{d}_lvl"] for d in reg_names["niche_dials"]]
    for a in sk:
        mask[a, ci["skillsk_lvl"]] = mask[ci["skillsk_lvl"], a] = True
        for b in sk:
            mask[a, b] = True
    for a in ni:
        for b in ni:
            mask[a, b] = True
    consider = ~np.eye(K, dtype=bool) & ~mask
    max_abs = float(np.nanmax(np.abs(corr[consider])))
    # top offending pairs
    pairs = []
    for a in range(K):
        for b in range(a + 1, K):
            if consider[a, b]:
                pairs.append((cols[a], cols[b], float(corr[a, b])))
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    return max_abs, pairs[:8], pd.DataFrame(corr, index=cols, columns=cols)


def _length_corr(df_bal: pd.DataFrame) -> tuple:
    out = {}
    w = df_bal["n_words"].to_numpy(float)
    for c in D.PRIMARY_LVL_COLS:
        v = df_bal[c].to_numpy(float)
        out[c] = float(np.corrcoef(w, v)[0, 1]) if v.std() > 0 else 0.0
    max_abs = max(abs(x) for x in out.values())
    return max_abs, out


def run_gates(df: pd.DataFrame, meta: dict, templates, job_ad, cfg: dict) -> dict:
    reg = D.build_registry(templates, job_ad)
    gen = cfg["generation"]
    corr_gate = float(gen["orthogonality"]["max_abs_corr"])
    len_gate = float(gen.get("length_corr_max", corr_gate))
    bal = df[~df["is_anchor"]].reset_index(drop=True)

    # 1. realized == assigned (clean text)
    mism = []
    for _, r in df.iterrows():
        probs = D.detect_mismatches(r["text_clean"], r["_profile"], reg)
        if probs:
            mism.append((r["letter_id"], probs))
    g_realized = len(mism) == 0

    # 2. validate() passes on every degraded letter
    val_fail = []
    for _, r in df.iterrows():
        probs = templates.validate(r["text"], [r["YEARS"], r["N_LANGUAGES"], r["N_LAUNCHES"]])
        if probs:
            val_fail.append((r["letter_id"], probs))
    g_validate = len(val_fail) == 0

    # 3. fluency: applied mistakes == target for every letter
    flu_bad = df[df["mistakes_applied"] != df["fluency_count"]]["letter_id"].tolist()
    g_fluency = len(flu_bad) == 0

    # 4. orthogonality (non-structural) on the balanced subset
    max_corr, top_pairs, corr_df = _orthogonality(bal, meta["registry"])
    g_ortho = max_corr < corr_gate

    # 5. length not correlated with any primary dial
    max_len_corr, len_corrs = _length_corr(bal)
    g_length = max_len_corr < len_gate

    # Not gates: the same two statistics over the FULL corpus, where the 16 anchor letters
    # enter. The paper quotes both (balanced 0.146 / 0.202; full corpus 0.195 / 0.219).
    full_corr, _, _ = _orthogonality(df.reset_index(drop=True), meta["registry"])
    full_len, _ = _length_corr(df.reset_index(drop=True))

    # 6. balance: each primary dial's levels within +/-1 of n/3 on the balanced set
    bal_report = {}
    ok_balance = True
    for c in D.PRIMARY_LVL_COLS:
        counts = bal[c].value_counts().to_dict()
        bal_report[c] = {int(k): int(v) for k, v in counts.items()}
        target = len(bal) / 3
        if any(abs(v - target) > 1 for v in counts.values()) or len(counts) != 3:
            ok_balance = False
    g_balance = ok_balance

    gates = {
        "realized_equals_assigned": g_realized,
        "validate": g_validate,
        "fluency_count": g_fluency,
        "orthogonality": g_ortho,
        "length_decorrelated": g_length,
        "balance": g_balance,
    }
    details = {
        "mismatches": mism[:10],
        "validate_failures": val_fail[:10],
        "fluency_bad": flu_bad[:10],
        "max_nonstructural_abs_corr": max_corr,
        "top_corr_pairs": top_pairs,
        "corr_df": corr_df,
        "max_length_corr": max_len_corr,
        "length_corrs": len_corrs,
        "balance": bal_report,
        "corr_gate": corr_gate,
        "len_gate": len_gate,
        "full_corpus_max_nonstructural_abs_corr": full_corr,
        "full_corpus_max_length_corr": full_len,
    }
    return {"gates": gates, "all_ok": all(gates.values()), "details": details}
