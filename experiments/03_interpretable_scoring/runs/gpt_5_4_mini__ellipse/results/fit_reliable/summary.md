# Fit summary — `interpretable_scoring_gpt_5_4_mini__ellipse`

- judge: `gpt_5_4_mini` (gpt-5.4-mini-2026-03-17)
- dataset: `ellipse`
- prompt variant: `comparative_verdict_qualities`
- n essays: 100
- K active criteria: 9
- T comparisons: 2000

## Global parameters

- β̂ = -1.0775 (SE 0.0685) ; σ(β̂) = 0.2540
- NLL (regularized) = 8070.0381 ; NLL (unregularized) = 8069.6631
- L-BFGS-B iterations: 132 ; converged: True
- Convex sanity: |L_primary − L_sanity| = 2.582e-05

## Per-criterion table

| k | canonical_name | share | std(s_k) | std(s_k) SE | γ_k | γ_k SE | sign_corr |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | Sentence Formation | 0.124 | 0.747 | 0.048 | -0.086 | 0.082 | +0.744 |
| 1 | Vocabulary | 0.121 | 1.010 | 0.072 | -0.714 | 0.087 | +0.934 |
| 2 | Cohesion and Transitions | 0.086 | 0.747 | 0.054 | -1.138 | 0.090 | +0.903 |
| 3 | Sentence Variety and Complexity | 0.056 | 0.994 | 0.058 | -1.695 | 0.112 | +0.707 |
| 4 | Clarity | 0.042 | 0.896 | 0.055 | -2.088 | 0.122 | +0.728 |
| 5 | Mechanics and Spelling | 0.038 | 0.983 | 0.076 | -2.766 | 0.149 | +0.873 |
| 6 | Academic Register | 0.029 | 1.091 | 0.070 | -3.146 | 0.159 | +0.830 |
| 7 | Focus and Relevance | 0.020 | 0.697 | 0.053 | -2.605 | 0.160 | +0.630 |
| 8 | Reasoning and Counterargument | 0.020 | 0.965 | 0.075 | -3.404 | 0.196 | +0.820 |

## Held-out validation

- n_train = 1600  n_test = 400
- BT-predictor winner accuracy = 0.9175
