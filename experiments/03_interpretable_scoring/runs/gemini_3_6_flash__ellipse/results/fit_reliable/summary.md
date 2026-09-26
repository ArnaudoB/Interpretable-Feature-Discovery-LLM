# Fit summary — `interpretable_scoring_gemini_3_6_flash__ellipse`

- judge: `gemini_3_6_flash` (gemini-3.6-flash)
- dataset: `ellipse`
- prompt variant: `comparative_verdict_qualities`
- n essays: 100
- K active criteria: 9
- T comparisons: 2000

## Global parameters

- β̂ = +0.3994 (SE 0.0836) ; σ(β̂) = 0.5985
- NLL (regularized) = 7334.6674 ; NLL (unregularized) = 7333.9813
- L-BFGS-B iterations: 195 ; converged: True
- Convex sanity: |L_primary − L_sanity| = 2.537e-05

## Per-criterion table

| k | canonical_name | share | std(s_k) | std(s_k) SE | γ_k | γ_k SE | sign_corr |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | Overall Organization | 0.167 | 1.200 | 0.067 | -0.682 | 0.094 | +0.914 |
| 1 | Vocabulary Range and Precision | 0.165 | 1.345 | 0.081 | -0.855 | 0.097 | +0.936 |
| 2 | Elaboration and Support | 0.121 | 1.153 | 0.059 | -1.154 | 0.103 | +0.809 |
| 3 | Sentence Variety and Complexity | 0.115 | 1.126 | 0.066 | -1.339 | 0.102 | +0.802 |
| 4 | Grammatical Accuracy | 0.086 | 1.515 | 0.073 | -2.471 | 0.134 | +0.796 |
| 5 | Transitions and Cohesive Devices | 0.040 | 0.830 | 0.074 | -2.774 | 0.149 | +0.871 |
| 6 | Paragraph Organization | 0.039 | 0.737 | 0.062 | -2.454 | 0.146 | +0.726 |
| 7 | Mechanics and Conventions | 0.031 | 1.575 | 0.109 | -4.536 | 0.234 | +0.876 |
| 8 | Topic Focus | 0.013 | 1.366 | 0.123 | -4.625 | 0.354 | +0.659 |

## Held-out validation

- n_train = 1600  n_test = 400
- BT-predictor winner accuracy = 0.9400
