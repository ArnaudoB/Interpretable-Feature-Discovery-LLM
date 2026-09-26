# Prior-only elicitation, open count

Cell for the open-count variant of the prior-only (P) elicitation: gpt-5.4 (high reasoning),
no data shown, asked to choose the number of dimensions itself (prompt:
`prompts/prior_open.py`). The P set used everywhere else (`feature_prior_root`) asks for exactly
16; this cell measures how many the model names when left free. It backs the open-count figures
of App. `app:methods-cmv` (recomputed by `scripts/14_paper_numbers.py`, record
`prior.open_count`). Nothing in this cell was scored.

## Prompt

Identical to the 16-feature prompt except for the task paragraph and the user turn (which drops
"the 16"):

> Your task. Name the dimensions for assessing how persuasive such a reply is. Give as many as the
> task warrants — decide the number yourself, and be thorough rather than selective. Do not pad the
> list either: add a dimension only when it names something none of the others already capture.

No target number is given (naming one makes the model pad the list to reach it), and the breadth
instruction concerns the list only, never the data. The schema does not fix the array length; the
parser accepts any count above a floor of 5.

## Counts per draw

Two independent runs of 10 draws each.

| run | counts | mean | sd |
|---|---|---:|---:|
| run 1 | 14, 16, 16, 18, 15, 17, 13, 16, 14, 18 | 15.7 | 1.7 |
| run 2 (canonical) | 15, 16, 15, 17, 16, 15, 17, 14, 14, 17 | 15.6 | 1.2 |

Pooled over both runs: 15.65 ± 1.42 (range 13–18).

## Label stability

Across run 2's 156 criteria there are 92 distinct names (82 ignoring case), none present in all
10 draws, and the pairwise exact-name Jaccard between draws is 0.00–0.29; much of the variation is
renaming (*evidence quality / evidential support / evidential grounding*). Mean best-match cosine
between draws (text-embedding-3-large, `results/compare_blind.json`) is 0.69–0.81; against the
discovered set it is 0.43–0.50, with no criterion above 0.80 in any draw. Cosine is a coarse
measure here: it does not represent antonymy (a criterion and its negation sit close together),
and short definitions compress its range.

## Files

| file | contents |
|---|---|
| `results/panels_blind.json` | run 2, all 10 draws with usage (canonical) |
| `results/panels_blind_run2.json` | identical copy of the above |
| `results/compare_blind.json` | pairwise cosines, draw vs discovered set, draw 0's nearest neighbours |
| `features_blind.json` | draw 0 of run 2 (15 criteria), promoted by position |
| `cache/` | embedding vectors |

Run 1 is recorded by its per-draw counts (the table above).
