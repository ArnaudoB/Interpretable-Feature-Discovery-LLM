# Interpretable Feature Discovery from Unstructured Data Using LLMs and Statistical Choice Models

Code and data accompanying the paper of the same title (Anonymous authors, under review).

## Method in one paragraph

Items (cover letters, online debate replies, essays) are compared in pairs along a sparse,
balanced comparison graph. For each pair, an LLM judge states in free text the features on
which the two items differ and which item is stronger on each; these **rationales are treated
as data**. A second LLM call groups the raw feature phrases into K shared features, giving a
verdict tensor `T[p, k] ∈ {-1, 0, +1}` (not cited / cited in favour of the first or second
item). A **statistical choice model** in the Bradley–Terry family then explains, for every
pair and feature, both *whether* the feature is cited (the **citation gate**,
`σ(log(2cosh Δ) + γ_k)`, driven by the size of the score gap `Δ = S[i,k] − S[j,k]` and a
per-feature citation propensity `γ_k`) and, when it is, *which* item wins (the **direction
channel**, `σ(Δ + β)`, with a shared position-bias term `β`). Penalised maximum likelihood
gives per-item, per-feature scores with sandwich standard errors. Two **reliability
diagnostics** flag features whose scores are not pinned down by the data: an identifiability
condition number `κ_k` and a signal-to-noise ratio `ρ_k`; the model is refit on the features
that pass both.

## Repository layout

```
core/                         the method, shared by all experiments
  graph/                      comparison graph (union of edge-disjoint perfect matchings)
  judging/                    request specs, content-addressed response cache, realtime and
                              batch clients (OpenAI, Anthropic, Gemini, DeepSeek), pricing
  canonical.py, tensors.py    phrase normalisation; verdict-tensor assembly
  fit.py                      design tensor, model fit, kappa/rho screen
  model/                      gated Bradley-Terry model (gated.py), diagnostics (diagnostics.py),
                              sandwich covariance (sandwich.py), derivation (DERIVATION.md)
    holistic/                 holistic-verdict variant used in the appendix extension
  metrics.py                  feature-recovery metrics (Experiment 1)
  evaluation/, analysis/      gate calibration; cross-judge consolidation (Experiment 3)
  cmv/                        ChangeMyView data, baselines, BT/pointwise scoring, paired task,
                              dimensionality, win rates (Experiment 2)
  embeddings.py, openai_client.py, ledger.py
experiments/
  01_synthetic_cover_letters/ Sec. 4 and App. C: feature recovery on a synthetic corpus
  02_changemyview/            Sec. 5 and App. D: ChangeMyView persuasion
  03_interpretable_scoring/   App. A: interpretable LLM-based scoring, 4 judges x 2 corpora
    each contains: README.md, config.yaml, scripts/ (numbered steps), prompts, figures/
    (plotting scripts and the figure PDFs), results/ (outputs), raw/ (LLM response caches
    and inputs)
tests/                        pytest suite
reproduce.sh                  runs the offline reproduction of every table and figure
figures/                      all paper figures, collected by reproduce.sh
```

## Installation

Python 3.12. From the repository root:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # pinned dependencies; see also requirements.txt
```

`make_cmv_win_rates.py` (Figure 3 and Figure 21) renders its labels with LaTeX and needs a TeX
installation (`latex`, `dvipng`); every other step is pure Python.

## Reproducing the paper

Every LLM response used in the paper is shipped in the experiments' response caches, keyed by
a hash of the exact request. The reproduction therefore needs **no API key and makes no API
call**:

```bash
./reproduce.sh all          # or: ./reproduce.sh exp1 | exp2 | exp3
pytest                      # regression tests on the regenerated results
```

`reproduce.sh` runs each experiment's offline steps in order (the per-experiment READMEs list
them) and copies every figure into `figures/`. Approximate runtimes on a laptop CPU:
Experiment 1 about 10 min, Experiment 2 about 1.5 h (dominated by the dimensionality
bootstrap, step 10), Experiment 3 about 20 min. Each experiment also writes
`results/paper_numbers.json`, which lists every number quoted in the text with its location in
the paper and the value computed from the artifacts.

| Paper artifact | Produced by (run from the experiment directory) | Output |
|---|---|---|
| Table 1 (`tab:main`) | 01: `scripts/09_build_tables_figures.py`, `scripts/09b_ablation_table.py`, `scripts/10_paper_tables.py` | `results/tables/tab_main.tex` |
| Tables 6, 7 (`tab:main-bands`, `tab:paired`) | 01: `scripts/08*_bootstrap*.py`, then `scripts/10_paper_tables.py` | `results/tables/tab_main_bands.tex`, `tab_paired.tex` |
| Table 4 (`tab:feats-ours`) | 01: `scripts/04_fit_comparative.py`, `scripts/10a_sandwich_ci.py` | `results/criterion_gamma_sandwich.csv` |
| Table 5 (`tab:feats-m1`) | 01: `scripts/05_elicit_rubrics.py full` (cached) | `results/rubric_stability/full_dataset_rubric.json` |
| Table 3, Figures 11–12 (`tab:dials`, `fig:job-ad`, `fig:template-skeleton`) | 01: corpus specification in `dataset/` | `dataset/{dials,job_ad,templates}.py` |
| Figures 13–14 (`fig:calibration_*_custom`) | 01: `scripts/12_calibration.py`, `figures/make_calibration_figures.py` | `figures/calibration_{reliability,skill}.pdf` |
| Figure 15 (`fig:paired`) | 01: `figures/make_paired_panels.py` | `figures/paired_{a,b}.pdf` |
| Figures 16–17 (`fig:recovery-heatmap-*`) | 01: `figures/make_recovery_heatmap_{comparative,pointwise}.py` | `figures/recovery_heatmap_{comparative,m1}.pdf` |
| Figure 18 (`fig:threshold_pair`) | 01: `figures/make_threshold_pair.py` | `figures/threshold_pair.pdf` |
| Tables 2, 9 (`tab:cmv-predictive`, `tab:cmv-predictive-2`) | 02: `scripts/09_paired_lr.py`, `scripts/13_paper_tables.py` | `results/tables/` |
| Figure 2 (`fig:cmv-dimension-curve`) | 02: `scripts/10_dimensionality.py`, `scripts/10b_matched_k.py`, `figures/make_cmv_dimension_accuracy.py` | `figures/cmv_dimension_accuracy.pdf` |
| Figure 20 (`fig:cmv-dimension-curve-full`) | 02: `scripts/10_dimensionality.py`, `figures/make_cmv_dimension_curve.py` | `figures/cmv_dimension_curve.pdf` |
| Figure 19 (`fig:cmv-diff-dr`) | 02: `scripts/10b_matched_k.py`, `figures/make_cmv_diff_grid.py` | `figures/cmv_diff_grid.pdf` |
| Figures 3, 21, Table 10 (`fig:cmv-win-rates*`, `tab:cmv-win-rates`) | 02: `scripts/11_win_rates.py`, `figures/make_cmv_win_rates.py [--full]`, `scripts/13_paper_tables.py` | `figures/cmv_win_rates{,_full}.pdf`, `results/tables/` |
| Table 8 (`tab:cmv-contamination`) | 02: `scripts/12_contamination.py`, `scripts/13_paper_tables.py` | `results/tables/` |
| Tables 11, 12 (`tab:cmv-features-p`, `tab:cmv-features-s`) | 02: `scripts/13_paper_tables.py` | `results/tables/` |
| Figure 4 (`fig:cross_judge_radar`) | 03: `scripts/07_cross_judge_cluster.py`, `figures/make_cross_judge_radar.py` | `figures/cross_judge_denoised_sigma.pdf` |
| Figure 5 (`fig:total_score_distribution`) | 03: `figures/make_total_score_distributions.py` | `figures/total_score_distributions.pdf` |
| Figures 6, 7 (`fig:radar_ellipse`, `fig:radar_asap`) | 03: steps 04–06, `figures/make_radar_panels.py` | `figures/radar_std_s_denoised_reliable{,_asap}_4judge_llm.pdf` |
| Figures 8–10 (`fig:calibration_*`, App. A) | 03: `scripts/08_calibration.py`, `figures/make_calibration_figures.py` | `figures/calibration_*_4judge_llm.pdf` |
| Numbers quoted in the text | 01: `scripts/13_claim_ledger.py`; 02: `scripts/14_paper_numbers.py`; 03: `scripts/09_paper_numbers.py` | `results/paper_numbers.json` |

Figure 1 is a schematic of the pipeline.

## Re-running the LLM stages (API access)

Every LLM stage can be re-issued against the providers' APIs. Copy `.env.example` to `.env` at
the repository root and fill in the keys you need. Stages that can spend money refuse to run
without `--yes`, print a cost projection first (`smoke` modes), stop at a per-run budget gate,
and write every response into the cache, so an interrupted run resumes at no extra cost. The
per-experiment READMEs list the paid commands.

| Experiment | Models | Mode | Approx. cost of a full re-run |
|---|---|---|---|
| 1 | GPT-5.4 mini (judge, scorer), GPT-5.4 high reasoning (feature grouping, rubrics), text-embedding-3-large | OpenAI Batch API | ~$110 at the rates of `core/judging/pricing.py` (≈$180 at current list prices) |
| 2 | GPT-5.4 mini (discovery, BT and pointwise scoring, zero-shot), GPT-5.4 high reasoning (feature grouping, P and S elicitation), text-embedding-3-large | OpenAI, realtime and Batch | ~$135 at the rates of `core/judging/pricing.py` (≈$335 at current list prices) |
| 3 | GPT-5.4 mini, Claude Haiku 4.5, Gemini 3.6 Flash, DeepSeek V4-Flash (judges); GPT-5.4 high reasoning (feature grouping, cross-judge consolidation) | OpenAI and DeepSeek realtime; Anthropic and Gemini Batch | ~$35 |

Costs are computed from the token usage recorded in the shipped caches. The pricing table in
`core/judging/pricing.py` is the one used for the costs reported in the paper; the config
`pricing:` section overrides it.

## Data

Included:

* **Synthetic cover letters (Experiment 1)**, generated deterministically by
  `experiments/01_synthetic_cover_letters/dataset/` (`scripts/00_generate_dataset.py`) from 16
  planted ground-truth features; the job advertisement they answer (`dataset/job_ad.py`) is a
  public job posting.
* **All LLM outputs** (judge responses, feature groupings, elicited rubrics and feature sets,
  pointwise scores), **embeddings** of feature descriptions, **fitted models** and **results**
  for the three experiments.
* **Third-party data**, used under the original terms, which continue to apply:
  * ChangeMyView *winning arguments* corpus (Tan et al., 2016, WWW), obtained through
    ConvoKit; `core/cmv/data.py` downloads it and `experiments/02_changemyview/raw/data/`
    holds the matched pairs used in the paper.
  * ELLIPSE corpus (Crossley et al., 2023), CC BY-NC-SA 4.0:
    `experiments/03_interpretable_scoring/raw/corpus/` (see `README_ELLIPSE.md` there).
  * ASAP 2.0 (Crossley et al., 2025), *Driverless cars* prompt: a 500-essay subset in the
    same directory, and the prompt's source article in `prompts/asap2.py`.

## The statistical model and diagnostics in code

* **Likelihood and fit**: `core/model/gated.py` (`nll`, `fit_model`). For pair `p = (i, j)`
  and feature `k`, with `Δ = S[i,k] − S[j,k]`: `P(cited) = σ(log(2cosh Δ) + γ_k)` and
  `P(first item favoured | cited) = σ(Δ + β)`. The negative log-likelihood plus a ridge
  penalty on the scores is minimised with L-BFGS-B; scores are centred per feature.
  `python -m core.model.gated --gradcheck` checks the analytic gradient.
* **Standard errors**: `core/model/sandwich.py` (`sandwich_V`, `gamma_beta_V`). The sandwich
  covariance `H⁻¹ B H⁻¹` uses the expected Fisher Hessian `H` and the per-comparison score
  outer products `B`, so errors are clustered by comparison.
* **κ (identifiability)**: `core/model/diagnostics.py` (`per_criterion_kappa`). The
  unpenalised score block of the Fisher information for feature `k` is a weighted Laplacian on
  the comparison graph; `κ_k = λ_max / λ_2` of that Laplacian, infinite if the feature's graph
  is disconnected.
* **ρ (reliability)**: `core/fit.py` (`kappa_rho`). `ρ_k = (Var ŝ_k − ν_k) / Var ŝ_k`, where
  `ν_k` is the mean sandwich variance of the fitted scores of feature `k`
  (`diagnostics.block_sandwich_s_diag`): the fraction of the observed score spread that is
  between-item signal rather than estimation noise.
* **Screen and refit**: features with `κ ≤ κ_max` and `ρ > ρ_min` are kept and the model is
  refit on them (`model.kappa_max`, `model.rho_threshold` in each `config.yaml`; e.g.
  `experiments/01_synthetic_cover_letters/scripts/04_fit_comparative.py`).
* **Holistic-verdict variant** (appendix extension): `core/model/holistic/` and its
  `DERIVATION.md`.

## Tests

```bash
pytest                 # add -m "not slow" to skip the sandwich-calibration test
```

The tests check the model's gradients and Hessian, that every LLM request rebuilt from the
code hits the shipped cache, and that the regenerated results match the values reported in
the paper.

## License

Code: MIT (see `LICENSE`). The synthetic corpus, LLM outputs, fitted models and results
generated for this work: CC BY 4.0. Third-party datasets remain under their original licenses
(see *Data*).
