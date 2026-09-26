# Cells — the runs behind the CMV numbers

Each directory is one scoring or elicitation run. A cell holds its `config.yaml` (the run's
parameters, with paths in the layout the run was executed in; `scripts/_config.py:source_path`
maps them here), its `results/` (score frames, fit report, and its comparison graph where it owns
one) and its `cache/` of content-addressed LLM responses, so every stage downstream of the API
replays offline. The experiment's `config.yaml` (`cells:` block) maps the paper's arms onto these
directories.

| cell | role |
|---|---|
| `feature_discovery_root` | discovery: 300 exchanges, 6,000 comparisons, taxonomy, gated fit |
| `feature_bt_root`, `feature_bt_root_dropped` | D + BT (the 16 gate-kept features, and the 13 dropped + 2 anchors) |
| `feature_bt_root_prior` | P + BT |
| `feature_bt_root_full` | graph and text baselines on all 3,411 training pairs (the `(full)` arms) |
| `pointwise_discovered`, `pointwise_dropped` | D + PW, run A |
| `pointwise_d29_rep_a`, `pointwise_d29_rep_b` | D + PW, run B (the 29 features reshuffled into two bundles) |
| `pointwise_prior`, `pointwise_prior_rep` | P + PW, runs A and B |
| `pointwise_sample`, `pointwise_sample_rep` | S + PW, runs A and B |
| `pointwise_discovered_full`, `pointwise_dropped_full` | the 29 features on the 3,411-pair graph (win rates) |
| `feature_prior_root` | the P elicitation (16 features, no data shown) |
| `feature_prior_root_open` | the open-count P elicitation (see its `REPORT.md`) |
| `feature_sample_root` | the S elicitation (42 features, data shown) |
| `contamination_probe_800` | the memorization probe |
| `zeroshot_pairwise` | the zero-shot baseline |
| `embedding_arms` | the text-embedding-3-large arms (summaries only, see below) |
| `matched_k` | per-subset fits of the matched-k analysis (not LLM output, see below) |

## Item ids

`item_id` (`wa%04d`) is assigned **positionally, per graph**. Two cells' ids denote the same
item only when they were built on the same graph, which `config.yaml`'s `graph_of` block
records and `_config.items_for()` enforces. The four D-pointwise cells share one graph, which
is what makes their two-run mean joinable; the `(full)` cells are built on 3,411 training
pairs and their ids mean something else entirely. Across graphs, join on `(pair_id, side)`.

## Notes on individual cells

* **`embedding_arms` ships without its cache.** The cache is a single 1.4 GB store of
  117,077 `text-embedding-3-large` vectors, needed only to re-derive the two embedding accuracies
  that the cell's `summary_n{400,3411}.parquet` and `pair_predictions_n*.parquet` already record.
  Re-running the embedding arm from scratch therefore needs the OpenAI API; it is the only arm
  of this experiment for which that is true.
* **`feature_discovery_root`**: `scores.parquet` and `criterion_gamma_sandwich.csv` hold the
  discovery fit used throughout; the D_r gate is applied to the latter. Of
  `criterion_diagnostics.parquet` and `fit_summary.json`, only `K_total` is read.
* **`pointwise_d29_rep_a`** holds 2,399 of its 2,400 exchanges: one cached response is a run
  of whitespace that hit the output-token cap and does not parse. The two-run mean
  (`core.cmv.paired.average_runs`) takes run A's score for that exchange.
* **`pointwise_sample`, `pointwise_sample_rep`**: gpt-5.4-mini occasionally returns a panel with
  a block of criteria missing. Such panels were re-requested with the identical request
  (`07_score_pointwise.py retry`); the replaced records are kept under `cache/responses/incomplete/`.
* **`matched_k/`** holds the per-subset L1-LR fits of the matched-k analysis (`fits/`), read by
  `scripts/10b_matched_k.py`; its `refit` mode recomputes them.
* **`feature_prior_root_open/REPORT.md`** records the per-draw feature counts of both open-count
  runs.
* **`feature_bt_root_full`** holds the graph shared by the (full) BOW / POS rows and the (full)
  pointwise cells.
