# Fit summary — `interpretable_scoring_gpt_5_4_mini__asap2`

- judge: `gpt_5_4_mini` (gpt-5.4-mini-2026-03-17)
- dataset: `asap2_driverless`
- prompt variant: `comparative_verdict_qualities_asap2_srcarticle`
- n essays: 100
- K active criteria: 18
- T comparisons: 2000

## Global parameters

- β̂ = -2.0772 (SE 0.1410) ; σ(β̂) = 0.1113
- NLL (regularized) = 11218.3652 ; NLL (unregularized) = 11216.5883
- L-BFGS-B iterations: 417 ; converged: True
- Convex sanity: |L_primary − L_sanity| = 1.853e-06

## Per-criterion table

| k | canonical_name | share | std(s_k) | std(s_k) SE | γ_k | γ_k SE | sign_corr |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | Organization | 0.179 | 1.375 | 0.096 | +0.815 | 0.091 | +0.897 |
| 1 | Use of source evidence | 0.165 | 1.360 | 0.074 | +0.709 | 0.092 | +0.786 |
| 2 | Thesis clarity and control | 0.146 | 1.086 | 0.067 | +0.720 | 0.085 | +0.420 |
| 3 | Elaboration | 0.127 | 0.699 | 0.047 | -0.043 | 0.074 | +0.716 |
| 4 | Conclusion | 0.084 | 0.620 | 0.045 | -0.946 | 0.078 | +0.803 |
| 5 | Counterargument handling | 0.071 | 1.302 | 0.075 | -1.264 | 0.103 | +0.849 |
| 6 | Analytical reasoning | 0.056 | 0.628 | 0.048 | -1.527 | 0.095 | +0.828 |
| 7 | Breadth of support | 0.039 | 0.551 | 0.056 | -1.609 | 0.097 | +0.499 |
| 8 | Argument depth | 0.027 | 0.950 | 0.075 | -2.928 | 0.157 | +0.817 |
| 9 | Coherence and cohesion | 0.023 | 0.750 | 0.072 | -2.840 | 0.153 | +0.751 |
| 10 | Sentence-level control | 0.016 | 1.064 | 0.104 | -3.892 | 0.239 | +0.758 |
| 11 | Nuanced stance | 0.015 | 1.493 | 0.109 | -3.931 | 0.237 | +0.798 |
| 12 | Specificity | 0.013 | 1.241 | 0.109 | -3.668 | 0.239 | +0.704 |
| 13 | Rhetorical force | 0.012 | 1.626 | 0.133 | -4.226 | 0.284 | +0.759 |
| 14 | Relevance | 0.011 | 1.464 | 0.139 | -4.127 | 0.317 | +0.730 |
| 15 | Introductory framing | 0.007 | 1.834 | 0.193 | -5.855 | 0.493 | +0.645 |
| 16 | Formal tone | 0.006 | 2.982 | 0.380 | -9.441 | 0.926 | +0.695 |
| 17 | Personal voice | 0.004 | 1.985 | 0.201 | -7.547 | 0.540 | +0.629 |

## Held-out validation

- n_train = 1600  n_test = 400
- BT-predictor winner accuracy = 0.9225
