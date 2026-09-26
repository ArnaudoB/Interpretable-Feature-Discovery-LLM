# Corpora

Both inputs ship here so step 00 regenerates each run's essay sample offline.

| File | Corpus | Used by |
|---|---|---|
| `ELLIPSE_Final_github_train.csv` | ELLIPSE (Crossley et al., 2023), training split; see `README_ELLIPSE.md` for the corpus's own description and citation. Licence: CC BY-NC-SA 4.0. | the four `*__ellipse` runs |
| `asap2_driverless_selected_items.parquet` | ASAP 2.0 (Crossley et al., 2025): a 500-essay subset of the *Driverless cars* prompt (md5 `7d7744d1e8e1314fde1d450ecb3cb3e1`) | the four `*__asap2` runs |

## The ASAP 2.0 subset

The subset is a fixed input to this experiment; it is not re-derived here.

* **Prompt.** All 500 essays answer the *Driverless cars* source-based argumentation task,
  so the ASAP runs are single-prompt (`prompt_name == "Driverless cars"`).
* **Provenance.** Every essay has `provenance == "fresh"`: it appears in neither the
  PERSUADE corpus nor the 2024 Kaggle release of ASAP 2.0, whose text and labels have been
  public since April 2024.
* **Balance.** 500 essays balanced over score × ELL status × length quartile at seed 42
  (score 1–6: 89/100/100/100/100/11).

Step 00 draws the 100-essay, band-balanced sample of every ASAP run from these 500, with the
holistic `score` (1–6) as the band, so `score_gap` spans 0–5 (ELLIPSE: 0–4). Only
`essay_id`, `full_text`, `score` and `prompt_name` are read; the other columns record how
the subset was built (match against the public releases, ELL status, length quartile,
selection seed and rank).

For the corpus itself, its citation and its terms of use, see the ASAP 2.0 release
(Crossley et al., 2025).
