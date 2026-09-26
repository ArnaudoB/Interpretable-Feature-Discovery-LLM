# Fit summary — `interpretable_scoring_deepseek_v4_flash__ellipse`

- judge: `deepseek_v4_flash` (deepseek-v4-flash)
- dataset: `ellipse`
- prompt variant: `comparative_verdict_qualities`
- n essays: 100
- K active criteria: 15
- T comparisons: 2000

## Global parameters

- β̂ = +1.2325 (SE 0.1185) ; σ(β̂) = 0.7743
- NLL (regularized) = 9012.4580 ; NLL (unregularized) = 9008.1942
- L-BFGS-B iterations: 1172 ; converged: True
- Convex sanity: |L_primary − L_sanity| = 5.837e-07

## Per-criterion table

| k | canonical_name | share | std(s_k) | std(s_k) SE | γ_k | γ_k SE | sign_corr |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | Organization and coherence | 0.204 | 3.839 | 0.261 | +2.916 | 0.219 | +0.513 |
| 1 | Vocabulary range and word choice | 0.146 | 1.522 | 0.080 | -0.135 | 0.096 | +0.835 |
| 2 | Syntactic complexity | 0.143 | 1.360 | 0.130 | +0.853 | 0.102 | +0.549 |
| 3 | Elaboration and support | 0.106 | 1.241 | 0.075 | -0.880 | 0.100 | +0.890 |
| 4 | Spelling, punctuation, and mechanics | 0.089 | 1.342 | 0.069 | -1.394 | 0.105 | +0.807 |
| 5 | Cohesion and transitions | 0.086 | 0.800 | 0.055 | -1.054 | 0.092 | +0.878 |
| 6 | Grammatical accuracy | 0.084 | 1.508 | 0.072 | -1.721 | 0.111 | +0.829 |
| 7 | Counterargument handling | 0.036 | 1.567 | 0.073 | -2.980 | 0.184 | +0.812 |
| 8 | Sentence formation | 0.030 | 1.541 | 0.093 | -2.365 | 0.151 | +0.823 |
| 9 | Rhetorical style and voice | 0.028 | 1.410 | 0.073 | -2.670 | 0.177 | +0.740 |
| 10 | Task awareness and register | 0.023 | 1.244 | 0.091 | -3.162 | 0.193 | +0.686 |
| 11 | Abstract reasoning | 0.009 | 1.582 | 0.135 | -4.247 | 0.387 | +0.586 |
| 12 | Source integration | 0.009 | 2.829 | 0.377 | -8.874 | 1.125 | +0.689 |
| 13 | Communicative clarity | 0.005 | 2.193 | 0.309 | -6.973 | 1.011 | +0.647 |
| 14 | English use consistency | 0.003 | 6.036 | 0.466 | -25.345 | 2.542 | +0.255 |

## Held-out validation

- n_train = 1600  n_test = 400
- BT-predictor winner accuracy = 0.9550
