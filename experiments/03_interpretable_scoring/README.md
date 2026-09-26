# Experiment 3 — interpretable LLM-based scoring, four judges × two corpora

Paper: Appendix `app:verdict-extension` (*Extension: Interpretable LLM-based scoring*) and its
sub-appendix `sec:extension-appendix`.

On each pair of essays the judge declares **which essay is stronger overall** and then cites
the qualities that essay has more of. The cited qualities are canonicalized into features,
and the holistic-verdict model ties the verdict to per-feature scores, so the overall
judgment decomposes into named, atomic features:

    P(w_p = 1)           = sigma( sum_k Delta_{p,k} + beta )
    P(c_{p,k} = 1 | w_p) = sigma( (2 w_p - 1) Delta_{p,k} + gamma_k )

This is not the gated model of the main text: the verdict logit is additive in the
per-feature gaps and the citation channel is driven by the signed margin in the winner's
favour. Code and derivation: `core/model/holistic/` (`DERIVATION.md`).

Four judges (GPT-5.4 mini, Claude Haiku 4.5, Gemini 3.6 Flash, DeepSeek V4-Flash) run the
identical pipeline on two corpora (ELLIPSE; ASAP 2.0 *Driverless cars*). **Within a corpus
only the judge varies**: same 100 essays, same 2,000 pairs, same slot order, the same prompt
byte for byte, the same canonicalizer (GPT-5.4, high reasoning). **Between corpora only the
corpus framing of the prompt varies.** The reliable features of the four judges are then
grouped into shared constructs (cross-judge consensus) by one more GPT-5.4 call per corpus,
and the calibration of the model's two gates is assessed out of fold.

Everything reproduces **offline**, with no API key: all 16,000 judge responses, the eight
canonicalization calls and the two consolidation calls are shipped, keyed by a hash of the
exact request.

---

## Data

`raw/corpus/` (details in `raw/corpus/README.md`):

| File | Corpus | Licence / citation |
|---|---|---|
| `ELLIPSE_Final_github_train.csv` | ELLIPSE training split (English-language-learner essays, grades 8–12, holistic + six analytic scores) | CC BY-NC-SA 4.0; Crossley et al. (2023), see `README_ELLIPSE.md` |
| `asap2_driverless_selected_items.parquet` | 500-essay subset of ASAP 2.0, *Driverless cars* prompt, essays absent from the public 2024 release | ASAP 2.0 terms of use; Crossley et al. (2025) |

Each run samples 100 band-balanced essays from its corpus (step 00).

---

## Layout

```
config.yaml                shared settings (graph, gates, model, validation) + per-corpus source
runs.yaml                  the 8 runs: judge, provider, corpus, token caps, spend gates
prompts/ellipse.py         elicitation + taxonomy prompts, ELLIPSE   (Prompt prm:elicit-ellipse)
prompts/asap2.py           same, ASAP 2.0; inlines the source article (Prompt prm:elicit-asap)
prompts/cross_judge.py     consolidation prompt + corpus framings    (Prompt prm:cross-judge)
dataset/                   ELLIPSE + ASAP 2.0 loaders, the band-balanced selector
scripts/                   steps 00-09; _config.py resolves every path, _paths.py sets sys.path
raw/corpus/                both corpora
raw/cache/<run>/           2,000 content-addressed judge responses per run (71 MB in all)
raw/cross_judge/           the two consolidation calls, request + verbatim output
raw/embed_cache/           81 feature embeddings; only fix the per-corpus radars' axis order
runs/<run>/results/        graph/, judgments, phrases, taxonomy/, pipeline{,_reliable}/,
                           fit/, fit_reliable/, cost_ledger.json
results/                   cross-run: cross_judge/, calibration/, total_score_agreement.csv,
                           paper_numbers.json
figures/                   the seven figures (PDF + PNG) and their scripts
```

The eight runs are `<judge>__<corpus>` with judge in `gpt_5_4_mini`, `claude_haiku_4_5`,
`gemini_3_6_flash`, `deepseek_v4_flash` and corpus in `ellipse`, `asap2`. Steps 00–06 take
`--run <name>`, a comma list, or `all`.

`runs/*/results/fit/cov.npy` (the all-feature sandwich covariances, 223 MB) is **not
shipped**: step 04 regenerates it and only step 05 reads it. `fit_reliable/cov.npy` is
shipped (71 MB); the figures and step 09 need it.

---

## Steps

Run from this directory. "API?" is whether the step can call a paid API; in the offline path
none does.

| Step | Command | Produces | Paper artifact | API? | Runtime |
|---|---|---|---|---|---|
| 00 | `scripts/00_build_pairs.py --run all` | `runs/<run>/results/graph/` (essays, 2,000 pairs, build report) | design, `sec:extension-appendix` | no | < 1 min |
| 01 | `scripts/01_judge.py collect --run all` | `judgments.parquet`, `phrases.parquet`, `cost_ledger.json` from `raw/cache/` | elicitation, `prm:elicit-ellipse`, `prm:elicit-asap` | live modes only | < 1 min |
| 02 | `scripts/02_taxonomy.py --replay --run all` | re-derives and checks `taxonomy/` against the shipped raw responses | canonicalization, `sec:extension-appendix` | live mode only | < 1 min |
| 03 | `scripts/03_build_tensors.py --run all` | `pipeline/` (tensors, active features) | — | no | < 1 min |
| 04 | `scripts/04_fit.py --run all` | `fit/` (all-feature fit, `cov.npy`) | — | no | ~10 min |
| 05 | `scripts/05_diagnostics.py --run all` | `fit/criterion_diagnostics.parquet`, `pipeline_reliable/` | reliability screen, `sec:extension-appendix` | no | < 1 min |
| 06 | `scripts/06_fit_reliable.py --run all` | `fit_reliable/` (the fit every number comes from) | all of `app:verdict-extension` | no | < 5 min |
| 07 | `scripts/07_cross_judge_cluster.py --corpus all` | `results/cross_judge/` (replays the two consolidation calls) | `prm:cross-judge`, cross-judge consolidation | `--live` only | < 1 min |
| 08 | `scripts/08_calibration.py` | `results/calibration/` (ECE, Brier skill, bootstrap CIs) | model calibration, `sec:extension-appendix` | no | ~5 min |
| fig | `figures/make_cross_judge_radar.py` | `figures/cross_judge_denoised_sigma.pdf` | `fig:cross_judge_radar` | no | < 1 min |
| fig | `figures/make_radar_panels.py` | `figures/radar_std_s_denoised_reliable{,_asap}_4judge_llm.pdf` | `fig:radar_ellipse`, `fig:radar_asap` | no (embedding cache) | < 1 min |
| fig | `figures/make_total_score_distributions.py` | `figures/total_score_distributions.pdf`, `results/total_score_agreement.csv` | `fig:total_score_distribution` | no | < 1 min |
| fig | `figures/make_calibration_figures.py` | `figures/calibration_{reliability,reliability_asap,brier_skill}_4judge_llm.pdf` | `fig:calibration_reliability_ellipse`, `fig:calibration_reliability_asap`, `fig:calibration_skill` | no | < 1 min |
| 09 | `scripts/09_paper_numbers.py` | `results/paper_numbers.json` | numbers quoted in `app:verdict-extension` | no | < 1 min |

### Offline path

```bash
cd experiments/03_interpretable_scoring
python scripts/00_build_pairs.py         --run all
python scripts/01_judge.py collect       --run all
python scripts/02_taxonomy.py --replay   --run all
python scripts/03_build_tensors.py       --run all
python scripts/04_fit.py                 --run all     # ~10 min (sandwich covariances)
python scripts/05_diagnostics.py         --run all
python scripts/06_fit_reliable.py        --run all
python scripts/07_cross_judge_cluster.py --corpus all
python scripts/08_calibration.py                       # ~5 min
python figures/make_cross_judge_radar.py
python figures/make_radar_panels.py
python figures/make_total_score_distributions.py      # also writes results/total_score_agreement.csv
python figures/make_calibration_figures.py
python scripts/09_paper_numbers.py
```

Since `fit_reliable/`, `results/` and the figures are shipped, the figures and step 09 can
also be run directly without steps 00–08.

### Paid path

Re-issuing the LLM calls requires keys in `.env` at the repository root (see `.env.example`).
Stages that spend refuse to run without `--yes`, abort above the run's `budget_gate_usd`, and
take one run (or one corpus) at a time. Responses are cached, so re-runs cost nothing.

| Call | Command | Provider | Key |
|---|---|---|---|
| GPT-5.4 mini judge | `01_judge.py sync --run gpt_5_4_mini__<corpus> --yes` | OpenAI (Responses API) | `OPENAI_API_KEY` |
| Claude Haiku 4.5 judge | `01_judge.py batch --run claude_haiku_4_5__<corpus> --yes` | Anthropic (Message Batches) | `ANTHROPIC_API_KEY` |
| Gemini 3.6 Flash judge | `01_judge.py gemini --run gemini_3_6_flash__<corpus> --yes` | Google (Gemini Batch API) | `GEMINI_API_KEY` |
| DeepSeek V4-Flash judge | `01_judge.py sync --run deepseek_v4_flash__<corpus> --yes` | DeepSeek (OpenAI-compatible API) | `DEEPSEEK_API_KEY` |
| canonicalization (GPT-5.4, high) | `02_taxonomy.py --run <run> --yes` | OpenAI | `OPENAI_API_KEY` |
| cross-judge consolidation (GPT-5.4, high) | `07_cross_judge_cluster.py --corpus <corpus> --live --yes` | OpenAI | `OPENAI_API_KEY` |

`01_judge.py smoke --run <run>` sends a few pairs and prints the projected cost first.
Reasoning is off on every judge: `reasoning_effort: none` (GPT), no extended thinking
(Haiku), `thinking_level: MINIMAL` (Gemini), `thinking: disabled` (DeepSeek).

---

## The eight runs

| Run | Judge API (mode) | Phrases → taxonomy | Features → reliable | β̂ (reliable) | Held-out verdict acc. |
|---|---|---|---:|---:|---:|
| `gpt_5_4_mini__ellipse` | openai (sync) | 267 → 18 | 17 → 9 | −1.078 | 0.917 |
| `claude_haiku_4_5__ellipse` | anthropic (batch) | 438 → 26 | 26 → 9 | −0.457 | 0.938 |
| `gemini_3_6_flash__ellipse` | gemini (batch) | 316 → 24 | 23 → 9 | +0.399 | 0.940 |
| `deepseek_v4_flash__ellipse` | deepseek (sync) | 374 → 16 | 15 → 7 | +0.594 | 0.948 |
| `gpt_5_4_mini__asap2` | openai (sync) | 324 → 19 | 18 → 15 | −2.032 | 0.922 |
| `claude_haiku_4_5__asap2` | anthropic (batch) | 568 → 19 | 18 → 16 | +0.536 | 0.935 |
| `gemini_3_6_flash__asap2` | gemini (batch) | 327 → 17 | 17 → 6 | +0.831 | 0.910 |
| `deepseek_v4_flash__asap2` | deepseek (sync) | 506 → 14 | 14 → 10 | +2.599 | 0.940 |

"Phrases" are normalized phrases cited at least 3 times, which are what the canonicalizer
sees. The taxonomy count includes the catch-all "Other", which is dropped. The reliability
screen keeps features with ρ > 0.50 and κ < 100 (`config.yaml`).

---

## Paper numbers

`results/paper_numbers.json` holds one record per number (or group of numbers) quoted in the
appendix, recomputed from the shipped artifacts by step 09: `id`, `where` (paper label),
`what`, `value`. Record groups: `design.*`, `elicitation.parsed`, `gates.*` (screen,
κ vs ρ removals, gate sensitivity at ρ > 0.70, κ < 50), `denoising.ellipse`, `sigma.*`
(cross-judge spreads), `agreement.spearman.*`, `position_bias`, `calibration.*`.
