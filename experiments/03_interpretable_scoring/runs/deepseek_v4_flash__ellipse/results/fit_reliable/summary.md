# Fit summary — `interpretable_scoring_deepseek_v4_flash__ellipse`

- judge: `deepseek_v4_flash` (deepseek-v4-flash)
- dataset: `ellipse`
- prompt variant: `comparative_verdict_qualities`
- n essays: 100
- K active criteria: 7
- T comparisons: 2000

## Global parameters

- β̂ = +0.5937 (SE 0.0680) ; σ(β̂) = 0.6442
- NLL (regularized) = 6374.0079 ; NLL (unregularized) = 6373.3816
- L-BFGS-B iterations: 158 ; converged: True
- Convex sanity: |L_primary − L_sanity| = 3.210e-05

## Per-criterion table

| k | canonical_name | share | std(s_k) | std(s_k) SE | γ_k | γ_k SE | sign_corr |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | Syntactic complexity | 0.143 | 1.567 | 0.134 | +0.645 | 0.104 | +0.633 |
| 1 | Elaboration and support | 0.106 | 1.392 | 0.082 | -1.138 | 0.105 | +0.911 |
| 2 | Spelling, punctuation, and mechanics | 0.089 | 1.332 | 0.069 | -1.550 | 0.101 | +0.860 |
| 3 | Cohesion and transitions | 0.086 | 0.931 | 0.061 | -1.279 | 0.097 | +0.898 |
| 4 | Grammatical accuracy | 0.084 | 1.462 | 0.070 | -1.846 | 0.108 | +0.893 |
| 5 | Sentence formation | 0.030 | 1.337 | 0.074 | -2.613 | 0.147 | +0.814 |
| 6 | Task awareness and register | 0.023 | 1.251 | 0.088 | -3.623 | 0.211 | +0.777 |

## Held-out validation

- n_train = 1600  n_test = 400
- BT-predictor winner accuracy = 0.9475
