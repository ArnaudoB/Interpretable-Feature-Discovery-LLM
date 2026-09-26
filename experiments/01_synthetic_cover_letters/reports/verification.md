# Cover-letter corpus — dataset verification

Synthetic corpus: **N = 200** (184 balanced + 16 extreme anchors), seed 42. Pure templating + deterministic fluency degradation, no LLM.

## Gates: ALL GATES PASS

| gate | result |
|---|---|
| 1. realized == assigned | pass (0 shown) |
| 2. validate | pass (0 shown) |
| 3. fluency-count | pass |
| 4. orthogonality | pass (max |corr| = 0.146 < 0.15) |
| 5. length-decorrelated | pass (max |corr| = 0.202 < 0.22) |

Over the full corpus (anchors included; reported, not gated): max non-structural |corr| = 0.195, max length |corr| = 0.219.

| 6. balance | pass |

## Dial balance (balanced subset)

| dial | level counts |
|---|---|
| exp_lvl | {1: 62, 0: 61, 2: 61} |
| edu_lvl | {0: 62, 2: 61, 1: 61} |
| school_lvl | {1: 62, 0: 61, 2: 61} |
| employer_lvl | {1: 62, 2: 61, 0: 61} |
| nlang_lvl | {1: 62, 2: 61, 0: 61} |
| nlaunch_lvl | {0: 62, 2: 61, 1: 61} |
| skillsk_lvl | {2: 62, 0: 61, 1: 61} |
| fluency_lvl | {1: 62, 0: 61, 2: 61} |

## Orthogonality — top non-structural pairs

| dial_a | dial_b | corr |
|---|---|---|
| skill_vuln_triage_automation | niche_side_project_lvl | +0.146 |
| exp_lvl | niche_side_project_lvl | +0.146 |
| exp_lvl | skill_dsa_depth | -0.142 |
| edu_lvl | skill_ai_agents_experience | -0.132 |
| fluency_lvl | skill_dsa_depth | +0.128 |
| nlaunch_lvl | skill_python | +0.122 |
| nlaunch_lvl | niche_open_source_lvl | -0.117 |
| edu_lvl | niche_ctf_lvl | +0.107 |

_Structural correlations (skill-vs-k, skill-vs-skill, niche-vs-niche) are excluded by design; full matrix in `orthogonality_corr.csv`._

## Length vs dials (balanced subset)

| dial | corr(word_count, level) |
|---|---|
| exp_lvl | +0.146 |
| edu_lvl | -0.068 |
| school_lvl | +0.055 |
| employer_lvl | -0.019 |
| nlang_lvl | -0.025 |
| nlaunch_lvl | -0.008 |
| skillsk_lvl | +0.202 |
| fluency_lvl | +0.051 |

Word-count range: 263-306 (mean 286).
