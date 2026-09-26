"""Frozen job ad for the cover-letter corpus.

Source: real Google posting, "Software Engineer, Information Security
Engineering" (Singapore), retrieved 2026-07-03.
Link: https://www.google.com/about/careers/applications/jobs/results/111928378944037574-software-engineer-information-security-engineering?category=DATA_CENTER_OPERATIONS&category=DEVELOPER_RELATIONS&category=HARDWARE_ENGINEERING&category=INFORMATION_TECHNOLOGY&category=MANUFACTURING_SUPPLY_CHAIN&category=NETWORK_ENGINEERING&category=PRODUCT_MANAGEMENT&category=PROGRAM_MANAGEMENT&category=SOFTWARE_ENGINEERING&category=TECHNICAL_INFRASTRUCTURE_ENGINEERING&category=TECHNICAL_SOLUTIONS&category=TECHNICAL_WRITING&category=USER_EXPERIENCE&employment_type=FULL_TIME&employment_type=PART_TIME&employment_type=TEMPORARY&hl=en_US&q=%22Software%20Engineer%22

Trimmed relative to the original posting (rule: trim boilerplate, never
qualifications): the Singapore right-to-work line (applicant-pool
statement, not a merit qualification; letters mention visa status in
zero cells, so absence is symmetric), the generic Google-SWE paragraph,
and the Core-team paragraph. All qualifications, role-specific
description, and responsibilities are verbatim.

Screening verdict (2026-07-03): passes the 5-point clean-arm checklist.
(1) No communication/language qualification anywhere; no customer-facing
    self-description.
(2) Level headroom OK but posting is unleveled; the "3 years DS&A"
    preferred item signals ~L3/L4, so the experience dial is CAPPED at
    3/5/7 (dial-max must stay in-band; see overqualification trap).
(3) Six addressable skills items after splitting (below).
(4) Four easy mandatory slots; Go is a MINIMUM, hence a mandatory slot
    and deliberately excluded from the skills pool.
(5) Boilerplate cleanly separable (trimmed above).
Residual, by design decision: per-item salience is heterogeneous (the
AI-agents item is sexier than Python) -> item subsets at fixed k MUST be
assigned by balanced design; see SKILLS_SUBSET_ASSIGNMENT below.

DO NOT EDIT after the first judge call. Any change forks the corpus.
Version this file alongside the generator prompts.
"""

JOB_AD_ID = "google_swe_infosec_singapore_2026"

JOB_AD = """\
Software Engineer, Information Security Engineering
Google - Singapore

Minimum qualifications:
- Bachelor's degree or equivalent practical experience.
- 2 years of experience testing, maintaining, or launching software products.
- 2 years of experience with software development in one or more programming languages.
- Experience programming in Go.

Preferred qualifications:
- Master's degree or PhD in Computer Science or a related technical field.
- 3 years of experience with data structures and algorithms.
- Experience in Python.
- Experience with Artificial Intelligence.
- Ability to perform security assessments.

About the job:
As a Software Engineer, you will leverage your software development expertise to design, build, and implement agentic tooling and systems that will automate and scale security processes, analyze risks, detect vulnerabilities, and proactively assist developer teams across Google in embedding security best practices throughout the entire development lifecycle. You will be crucial in enabling Google to respond to incoming vulnerability reports and remediate security issues.

Responsibilities:
- Collaborate closely with information security engineers to develop new tools for product security consulting, response and research, especially (but not exclusively) related to AI agents.
- Steer the strategic development of agents and infrastructure to automate vulnerability handling, triage, and conduct variant analysis to reduce manual workloads.
- Design and implementation of Go back-end systems and Python AI agents.
"""

# ---------------------------------------------------------------------------
# Design anchors derived from this ad. Kept next to the ad text so the ad
# and the design it induces cannot silently drift apart.
# ---------------------------------------------------------------------------

# Every profile MUST satisfy every minimum qualification (knockout guard):
# mandatory slot content in every letter. Go lives HERE, not in the skills
# pool (it is a minimum; putting it in the pool would blur the
# minimum/preferred boundary).
MINIMUM_QUALS = [
    "bachelors_or_equivalent",
    "2y_testing_maintaining_launching",  # collapsible with the next into one sentence
    "2y_software_development",
    "go_programming",
]

# Experience dial: capped for level headroom (unleveled posting; the
# "3 years DS&A" preferred item anchors the target band at ~L3/L4).
# Dial-max 7 reads "strong senior-competitive candidate"; do NOT extend
# upward or the overqualification trap re-enters through the side door.
EXPERIENCE_DIAL_YEARS = [3, 5, 7]

# Education dial. Upper end is validated by the ad's own text (MS/PhD is
# an explicit preferred qualification) -- judge credit for PhD follows
# the instrument, not a prior. Dominance ordering holds within-field.
EDUCATION_DIAL = ["BS", "MS", "PhD"]  # CS or stated related field

# Skills-match pool: preferred quals + responsibilities split into six
# letter-addressable items. Education excluded (own dial); Go excluded
# (minimum qualification, mandatory slot).
# CAUTION on item 1: the ad phrases DS&A with a year count ("3 years");
# the letter slot must express DEPTH WITHOUT YEARS (concrete artifact,
# no duration) or it entangles arithmetically with the experience dial.
SKILLS_POOL = [
    "dsa_depth",  # data structures & algorithms -- de-yeared phrasing only
    "python",
    "ai_agents_experience",  # high-salience item; see balanced assignment below
    "security_assessments",
    "vuln_triage_automation",  # from responsibilities: vulnerability handling/triage/variant analysis
    "agentic_tooling_design",  # from responsibilities: agentic tooling, Go back-ends + Python agents
]
SKILLS_MATCH_DIAL_K = [2, 4, 6]

# ---------------------------------------------------------------------------
# Numbered auxiliary dials. Both are self-contained COUNTS (never a
# duration), each realized as a single digit in a FIXED background clause that
# is present in EVERY letter -> they are graded numbered dials, not presence
# signals, and add no length variation across their levels. Together with
# EXPERIENCE_DIAL_YEARS they are the entire digit-validator whitelist. Assigned
# ORTHOGONALLY to the experience dial (a count is not a duration): n_launches is
# deliberately decoupled from years so it cannot re-enter as an experience proxy.
# ---------------------------------------------------------------------------
N_LANGUAGES_DIAL = [2, 5, 9]  # "Go plus N other languages" (min qual: >=1 language)
N_LAUNCHES_DIAL = [2, 4, 6]  # "launched N production services" (min qual: launching)

# Per-item salience is heterogeneous (ai_agents_experience is the sexy
# citation; python is not). At fixed k, WHICH k items a letter addresses
# must be assigned by a balanced (incomplete block) design so that item
# identity is orthogonal to the k-dial across the corpus. This turns
# per-item citation-rate asymmetry from a confound into a finding (which
# requirements does the judge attend to -- a gate-channel estimand).
SKILLS_SUBSET_ASSIGNMENT = "balanced_incomplete_block"

# No communication/language qualification anywhere in this ad and no
# customer-facing self-description -> the fluency arm is a clean
# dominance-nuisance dial here: judge credit for prose quality has no
# listed requirement to launder itself through. This ad is the clean
# halo-leak arm.
FLUENCY_DIAL_ROLE = "dominance_nuisance"
