# Fit summary — `interpretable_scoring_gemini_3_6_flash__ellipse`

- judge: `gemini_3_6_flash` (gemini-3.6-flash)
- dataset: `ellipse`
- prompt variant: `comparative_verdict_qualities`
- n essays: 100
- K active criteria: 23
- T comparisons: 2000

## Global parameters

- β̂ = +1.0357 (SE 0.1821) ; σ(β̂) = 0.7380
- NLL (regularized) = 10032.9574 ; NLL (unregularized) = 10014.7690
- L-BFGS-B iterations: 2521 ; converged: True
- Convex sanity: |L_primary − L_sanity| = 9.631e-07

## Per-criterion table

| k | canonical_name | share | std(s_k) | std(s_k) SE | γ_k | γ_k SE | sign_corr |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | Overall Organization | 0.167 | 1.159 | 0.065 | -0.576 | 0.093 | +0.893 |
| 1 | Vocabulary Range and Precision | 0.165 | 1.270 | 0.077 | -0.737 | 0.096 | +0.920 |
| 2 | Elaboration and Support | 0.121 | 1.141 | 0.061 | -1.051 | 0.102 | +0.766 |
| 3 | Sentence Variety and Complexity | 0.115 | 1.077 | 0.062 | -1.232 | 0.100 | +0.766 |
| 4 | Grammatical Accuracy | 0.086 | 1.453 | 0.073 | -2.320 | 0.129 | +0.775 |
| 5 | Spelling and Orthography | 0.071 | 2.196 | 0.124 | -2.368 | 0.171 | +0.498 |
| 6 | Transitions and Cohesive Devices | 0.040 | 0.817 | 0.073 | -2.636 | 0.148 | +0.812 |
| 7 | Paragraph Organization | 0.039 | 0.767 | 0.064 | -2.268 | 0.140 | +0.566 |
| 8 | Coherence and Flow | 0.035 | 0.749 | 0.068 | -2.134 | 0.140 | +0.401 |
| 9 | Sentence Formation | 0.034 | 1.271 | 0.119 | -2.481 | 0.147 | +0.657 |
| 10 | Mechanics and Conventions | 0.031 | 1.592 | 0.119 | -4.476 | 0.251 | +0.836 |
| 11 | Sentence Boundary Control | 0.026 | 1.762 | 0.064 | -3.610 | 0.219 | +0.435 |
| 12 | Topic Focus | 0.013 | 1.476 | 0.127 | -4.033 | 0.356 | +0.727 |
| 13 | Counterargument Integration | 0.011 | 2.171 | 0.212 | -6.377 | 0.722 | +0.631 |
| 14 | Clarity of Expression | 0.008 | 2.034 | 0.214 | -5.990 | 0.677 | +0.702 |
| 15 | Argument Coherence | 0.007 | 1.905 | 0.217 | -5.206 | 0.555 | +0.608 |
| 16 | Idiomaticity | 0.007 | 2.326 | 0.283 | -6.485 | 0.945 | +0.458 |
| 17 | Narrative Development | 0.006 | 3.104 | 0.432 | -8.625 | 1.381 | +0.603 |
| 18 | Text Completeness | 0.006 | 10.246 | 0.641 | -37.416 | 2.535 | +0.576 |
| 19 | Register and Tone | 0.005 | 2.195 | 0.382 | -8.235 | 1.312 | +0.621 |
| 20 | Rhetorical Engagement | 0.004 | 2.234 | 0.290 | -7.154 | 0.866 | +0.519 |
| 21 | Descriptive Language | 0.004 | 8.731 | 0.699 | -30.969 | 2.669 | +0.480 |
| 22 | Target Language Use | 0.002 | 11.028 | 0.723 | -50.178 | 3.737 | +0.398 |

## Held-out validation

- n_train = 1600  n_test = 400
- BT-predictor winner accuracy = 0.9375
