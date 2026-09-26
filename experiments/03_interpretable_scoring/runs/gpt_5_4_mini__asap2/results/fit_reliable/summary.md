# Fit summary — `interpretable_scoring_gpt_5_4_mini__asap2`

- judge: `gpt_5_4_mini` (gpt-5.4-mini-2026-03-17)
- dataset: `asap2_driverless`
- prompt variant: `comparative_verdict_qualities_asap2_srcarticle`
- n essays: 100
- K active criteria: 15
- T comparisons: 2000

## Global parameters

- β̂ = -2.0320 (SE 0.1300) ; σ(β̂) = 0.1159
- NLL (regularized) = 9755.4774 ; NLL (unregularized) = 9754.0864
- L-BFGS-B iterations: 320 ; converged: True
- Convex sanity: |L_primary − L_sanity| = 3.757e-07

## Per-criterion table

| k | canonical_name | share | std(s_k) | std(s_k) SE | γ_k | γ_k SE | sign_corr |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | Organization | 0.179 | 1.394 | 0.102 | +0.805 | 0.091 | +0.896 |
| 1 | Use of source evidence | 0.165 | 1.335 | 0.070 | +0.692 | 0.091 | +0.795 |
| 2 | Thesis clarity and control | 0.146 | 1.095 | 0.072 | +0.711 | 0.085 | +0.426 |
| 3 | Elaboration | 0.127 | 0.709 | 0.048 | -0.050 | 0.074 | +0.719 |
| 4 | Conclusion | 0.084 | 0.623 | 0.045 | -0.953 | 0.078 | +0.805 |
| 5 | Counterargument handling | 0.071 | 1.284 | 0.073 | -1.266 | 0.103 | +0.854 |
| 6 | Analytical reasoning | 0.056 | 0.631 | 0.048 | -1.535 | 0.095 | +0.827 |
| 7 | Argument depth | 0.027 | 0.950 | 0.074 | -2.935 | 0.156 | +0.817 |
| 8 | Sentence-level control | 0.016 | 1.062 | 0.106 | -3.903 | 0.240 | +0.748 |
| 9 | Nuanced stance | 0.015 | 1.513 | 0.114 | -4.015 | 0.244 | +0.811 |
| 10 | Specificity | 0.013 | 1.282 | 0.113 | -3.776 | 0.257 | +0.721 |
| 11 | Rhetorical force | 0.012 | 1.628 | 0.132 | -4.348 | 0.300 | +0.768 |
| 12 | Relevance | 0.011 | 1.407 | 0.129 | -4.110 | 0.304 | +0.746 |
| 13 | Introductory framing | 0.007 | 1.737 | 0.170 | -5.727 | 0.433 | +0.651 |
| 14 | Formal tone | 0.006 | 2.519 | 0.242 | -8.332 | 0.595 | +0.711 |

## Held-out validation

- n_train = 1600  n_test = 400
- BT-predictor winner accuracy = 0.9225
