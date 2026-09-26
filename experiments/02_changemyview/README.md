# Experiment 2 — ChangeMyView

Application of the feature-discovery pipeline to a real persuasion task: the paired
ChangeMyView "winning arguments" task of Tan et al. (2016). Given two replies to the same post,
one of which earned a delta (Δ) from the original poster and one of which did not, predict which
one persuaded. The experiment asks whether features **discovered** by pairwise comparison
describe persuasion in human-readable terms while retaining enough signal to predict it.

Paper: Sec. `sec:cmv`, with details in App. `app:experiment-details-cmv` (dataset, memorization
probe, methods) and App. `app:additional-results` (additional measures, discovered features).

## What is compared

Feature sets (all scored by gpt-5.4-mini):

| set | features | origin |
|---|---:|---|
| **D** | 29 | discovered: 6,000 pairwise comparisons of 300 training exchanges, free-text dimensions grouped by one gpt-5.4 taxonomy call, OP-side criteria excluded |
| **D_r** | 16 | the subset of D that passes the reliability gate (κ ≤ 50, ρ > 0.75) of the gated model |
| **P** | 16 | named by gpt-5.4 from a two-sentence description of the task, no data shown |
| **S** | 42 | named by gpt-5.4 after reading 752 unlabelled exchange pairs |

Scoring protocols: **BT** — per-feature Bradley–Terry on a frozen comparison graph (800 training
exchanges in a 16-regular anchor graph; each of 1,600 test exchanges compared against 16
anchors); **PW** — one 0–10 rating per exchange and feature, averaged over two scoring runs.

Analyses: held-out accuracy of an L1 logistic regression on the within-pair score difference
(400 training pairs, 800 held-out pairs), against Tan et al.'s features, bag-of-words / POS,
text embeddings, reply length and a zero-shot judge; the noise-corrected effective
dimensionality of each feature set; accuracy at a matched number of features k; per-feature win
rates; and a memorization (contamination) probe of the judge.

## Data

The corpus is ConvoKit's Winning-Args corpus (Tan et al. 2016). `core/cmv/data.py` downloads it
(~350 MB, into the standard ConvoKit location) only when the derived table has to be rebuilt; the
derived pair table `raw/data/pair_units.parquet` (4,263 matched pairs: 3,456 training, 807
held-out) is shipped, so no step below needs the download.

## Layout

```
config.yaml        analysis settings; `cells:` maps each arm onto a cell under raw/cells/
pipeline.yaml      settings of the judging steps 00-08 that are not in a cell config
prompts/           every prompt sent to an LLM (discovery, taxonomy, P / S elicitation,
                   BT and pointwise scoring, zero-shot, contamination probe)
scripts/           numbered steps (below); _config.py, _judge.py, _paths.py are shared helpers
figures/           figure scripts, paper_style.py, and the rendered PDFs / PNGs
raw/data/          the derived pair table
raw/cells/         one directory per scoring / elicitation run, with its LLM response cache
                   (see raw/cells/README.md)
results/           everything steps 09-14 write
```

Library code lives in `core/cmv/` (data, items, graphs, BT, pair task, designs, dimensionality,
matched-k, win rates) and the shared `core/` packages (judging and caching, gated model).

## Steps

Run from this directory with the repository's Python environment. "API" marks steps that can
call a paid API; those stages refuse to run without `--yes`, and every one of them has an offline
replay (`collect` / `rederive`) that reads the shipped caches. Runtimes are wall-clock on a
multi-core workstation.

| step | command | produces | paper artifact | API? | runtime |
|---|---|---|---|---|---|
| 00 | `scripts/00_build_items.py` | discovery items (300 exchanges) | App. `app:methods-cmv` | no | seconds |
| 01 | `scripts/01_build_pairs.py` | comparison graphs of every cell | App. `app:methods-cmv` | no | ~25 s (+ ~5 min for `bt_full`) |
| 02 | `scripts/02_judge_discovery.py` | 6,000 discovery comparisons, cited phrases | App. `app:methods-cmv`, Prompt `prm:cmv-comp` | yes | ~5 s (replay) |
| 03 | `scripts/03_taxonomy.py` | 45 canonical criteria and the phrase mapping | Prompt `prm:cmv-taxo` | yes | ~1 s (replay) |
| 04 | `scripts/04_fit_discovery.py` | gated fit, κ/ρ, sandwich CIs, discovery scores | App. `app:methods-cmv` (D_r) | no | ~2 min |
| 05 | `scripts/05_elicit_prior.py` | the P feature set | Prompt `prm:cmv-prior-elicit`, `tab:cmv-features-p` | yes | ~1 s (replay) |
| 05b | `scripts/05b_elicit_sample.py` | the S feature set and its reshuffled repeat | Prompt `prm:cmv-sample-elicit`, `tab:cmv-features-s` | yes | ~2 s (replay) |
| 06 | `scripts/06_score_bt.py` | BT judgments and scores of the D / P cells | Prompt `prm:cmv-score-bt` | yes | ~3 min (replay) |
| 07 | `scripts/07_score_pointwise.py` | pointwise scores of the 10 PW cells | Prompt `prm:cmv-score-pw` | yes | ~30 s (replay) |
| 08 | `scripts/08_baselines.py text` / `zeroshot` / `embeddings` | Tan / BOW / POS / #words, zero-shot and embedding predictions | baseline rows of `tab:cmv-predictive{,-2}` | zero-shot, embeddings | ~3 min (replay; embeddings not replayable) |
| 09 | `scripts/09_paired_lr.py` | `results/accuracy.csv`, `results/predictions/` | `tab:cmv-predictive`, `tab:cmv-predictive-2`, λ in App. `app:methods-cmv` | no | ~20 s |
| 10 | `scripts/10_dimensionality.py` | `results/dimensionality/`, `results/dimensionality.json` | `fig:cmv-dimension-curve` (left), `fig:cmv-dimension-curve-full`, App. `app:additional-measures` | no | ~15–20 min |
| 10b | `scripts/10b_matched_k.py` | `results/matched_k/` | `fig:cmv-dimension-curve` (right), `fig:cmv-diff-dr`, Sec. `sec:cmv` | no | ~40 s (`refit`: minutes per set) |
| 11 | `scripts/11_win_rates.py` | `results/win_rates.csv`, `win_rates_meta.json` | `tab:cmv-win-rates`, `fig:cmv-win-rates{,-full}` | no | seconds |
| 12 | `scripts/12_contamination.py` | `results/contamination.json` | `tab:cmv-contamination`, App. `app:cmv-contamination` | no | seconds |
| 13 | `scripts/13_paper_tables.py` | `results/tables/*.tex` (6 tables) | `tab:cmv-predictive`, `tab:cmv-predictive-2`, `tab:cmv-contamination`, `tab:cmv-win-rates`, `tab:cmv-features-p`, `tab:cmv-features-s` | no | seconds |
| 14 | `scripts/14_paper_numbers.py` | `results/paper_numbers.json` | numbers quoted in the text | no | seconds |
| fig | `figures/make_cmv_dimension_accuracy.py` | `figures/cmv_dimension_accuracy.pdf` | `fig:cmv-dimension-curve` | no | seconds |
| fig | `figures/make_cmv_dimension_curve.py` | `figures/cmv_dimension_curve.pdf` | `fig:cmv-dimension-curve-full` | no | seconds |
| fig | `figures/make_cmv_diff_grid.py` | `figures/cmv_diff_grid.pdf` | `fig:cmv-diff-dr` | no | seconds |
| fig | `figures/make_cmv_win_rates.py` [`--full`] | `figures/cmv_win_rates{,_full}.pdf` | `fig:cmv-win-rates`, `fig:cmv-win-rates-full` | no | seconds |

`make_cmv_win_rates.py` renders with LaTeX and needs `latex` and `dvipng` on the PATH
(`--no-tex` gives an on-screen approximation only). The other figure scripts use matplotlib's
own text engine.

## Offline path (0 API calls)

Everything the paper reports is recomputed from the shipped cells, with no API key and no
network:

```
python scripts/09_paired_lr.py
python scripts/10_dimensionality.py            # the slow one, ~15-20 min
python scripts/10b_matched_k.py
python scripts/11_win_rates.py
python scripts/12_contamination.py
python scripts/13_paper_tables.py
python scripts/14_paper_numbers.py
python figures/make_cmv_dimension_accuracy.py
python figures/make_cmv_dimension_curve.py
python figures/make_cmv_diff_grid.py
python figures/make_cmv_win_rates.py
python figures/make_cmv_win_rates.py --full
```

The judging steps 00-08 can also be replayed offline: every LLM stage rebuilds its requests
exactly and reads the responses from the shipped caches. Nothing is written into `raw/cells/`
unless `--in-place` is given, so replay into a scratch directory and compare file by file:

```
D=scratch/cmv_replay
python scripts/00_build_items.py --out-dir $D
python scripts/01_build_pairs.py --out-dir $D --skip bt_full   # omit --skip for the slow 3,411-pair graph
python scripts/02_judge_discovery.py collect --out-dir $D
python scripts/03_taxonomy.py rederive --out-dir $D
python scripts/04_fit_discovery.py --out-dir $D
python scripts/05_elicit_prior.py rederive --out-dir $D
python scripts/05b_elicit_sample.py collect --out-dir $D
python scripts/06_score_bt.py collect --out-dir $D
python scripts/07_score_pointwise.py collect --out-dir $D
python scripts/08_baselines.py text --out-dir $D
python scripts/08_baselines.py zeroshot collect --out-dir $D
```

Two stages replay from their saved outputs rather than from the response cache:
`03_taxonomy.py rederive` takes the criteria saved in `taxonomy.json` and re-derives
`mapping.json`; `05_elicit_prior.py rederive` re-derives the promoted feature set from the saved
panels. The embedding arm's predictions are shipped and read by step 09; recomputing the
embeddings themselves (`08_baselines.py embeddings`) calls the embeddings API.
`tests/test_cmv.py` and `tests/test_cmv_replay.py` (run with `pytest` from the repository root)
check the analysis outputs and that every shipped request hits the cache.

## Full paid path

To regenerate the cells from the corpus, run the same steps with `--in-place` and the paid
stages (`smoke` to measure cost on a few requests, `submit` for the OpenAI Batch API, then
`collect`; `run` for the realtime calls of steps 03, 05, 05b and the zero-shot baseline; `embed`
for the embeddings), each with `--yes`, e.g.

```
python scripts/02_judge_discovery.py submit --in-place --yes
python scripts/02_judge_discovery.py collect --in-place
python scripts/03_taxonomy.py run --in-place --yes
```

`07_score_pointwise.py retry --in-place --yes` re-requests pointwise panels that come back
incomplete (`07_score_pointwise.py audit` lists them without calling the API). The API key is
read from the environment or from `.env` at the repository root. Which discovered criteria are
challenger-side, and the bundles of the D repeat run, are fixed in the cells' `features.json`
rather than re-derived.

## Determinism and shipped inputs

* **Accuracies.** liblinear shuffles coordinates, drawing from numpy's global RNG when no
  `random_state` is given. Step 09 seeds that RNG once (`task.liblinear_global_seed`) and fits
  the score arms in a fixed order (`_config.SCORE_ARMS`), as the reported fits did; changing
  the order or passing a per-fit `random_state` moves individual arms by about one pair
  (±0.125 pp). Step 08 re-seeds before every fit, so a baseline arm does not depend on which
  arms were fitted before it; the all-Tan arm's cross-validation is nearly tied between
  C = 0.03 and C = 31.6 and is sensitive to that RNG state (see the step's docstring). Step 09
  reads the shipped per-pair predictions of the baselines.
* **Matched-k fits.** `raw/cells/matched_k/fits/*.npz` are the per-subset fits (held-out
  correctness per drawn feature subset, and the subsets) consumed by step 10b. The default mode
  reads them and recomputes everything downstream exactly; `10b_matched_k.py refit --panel <set>`
  recomputes one set's fits and reports the agreement with the shipped ones without overwriting
  them (liblinear's global-RNG stream makes a refit agree up to about one held-out pair per fit).
* **Dimensionality.** Step 10 is seeded end to end (`--seed`, default 0) and deterministic.
