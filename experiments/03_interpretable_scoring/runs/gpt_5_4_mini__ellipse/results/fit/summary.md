# Fit summary — `interpretable_scoring_gpt_5_4_mini__ellipse`

- judge: `gpt_5_4_mini` (gpt-5.4-mini-2026-03-17)
- dataset: `ellipse`
- prompt variant: `comparative_verdict_qualities`
- n essays: 100
- K active criteria: 17
- T comparisons: 2000

## Global parameters

- β̂ = -2.0647 (SE 0.1512) ; σ(β̂) = 0.1126
- NLL (regularized) = 9935.7565 ; NLL (unregularized) = 9932.7125
- L-BFGS-B iterations: 1076 ; converged: True
- Convex sanity: |L_primary − L_sanity| = 3.631e-06

## Per-criterion table

| k | canonical_name | share | std(s_k) | std(s_k) SE | γ_k | γ_k SE | sign_corr |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | Organization | 0.212 | 3.426 | 0.292 | +3.461 | 0.311 | +0.687 |
| 1 | Development and Support | 0.198 | 2.014 | 0.144 | +1.939 | 0.136 | +0.738 |
| 2 | Sentence Formation | 0.124 | 0.703 | 0.043 | +0.078 | 0.081 | +0.615 |
| 3 | Vocabulary | 0.121 | 0.868 | 0.063 | -0.542 | 0.083 | +0.911 |
| 4 | Cohesion and Transitions | 0.086 | 0.678 | 0.050 | -0.977 | 0.088 | +0.843 |
| 5 | Sentence Variety and Complexity | 0.056 | 0.959 | 0.058 | -1.443 | 0.104 | +0.587 |
| 6 | Clarity | 0.042 | 0.869 | 0.053 | -1.790 | 0.110 | +0.613 |
| 7 | Mechanics and Spelling | 0.038 | 0.807 | 0.068 | -2.428 | 0.131 | +0.868 |
| 8 | Academic Register | 0.029 | 1.143 | 0.071 | -2.912 | 0.162 | +0.696 |
| 9 | Grammar and Usage | 0.027 | 0.695 | 0.059 | -2.472 | 0.135 | +0.574 |
| 10 | Focus and Relevance | 0.020 | 1.032 | 0.089 | -2.186 | 0.146 | +0.488 |
| 11 | Reasoning and Counterargument | 0.020 | 1.037 | 0.075 | -3.010 | 0.183 | +0.690 |
| 12 | Argumentative Stance | 0.009 | 1.146 | 0.125 | -4.352 | 0.316 | +0.724 |
| 13 | Voice and Audience Awareness | 0.007 | 1.592 | 0.186 | -5.184 | 0.488 | +0.649 |
| 14 | Expressive Language | 0.005 | 1.973 | 0.346 | -7.412 | 1.137 | +0.634 |
| 15 | Genre Convention | 0.003 | 2.997 | 0.325 | -9.497 | 1.209 | +0.563 |
| 16 | Idiomatic Phrasing | 0.001 | 4.517 | 0.426 | -16.002 | 1.431 | +0.455 |

## Held-out validation

- n_train = 1600  n_test = 400
- BT-predictor winner accuracy = 0.9325
