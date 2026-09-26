# Fit summary — `interpretable_scoring_deepseek_v4_flash__asap2`

- judge: `deepseek_v4_flash` (deepseek-v4-flash)
- dataset: `asap2_driverless`
- prompt variant: `comparative_verdict_qualities_asap2_srcarticle`
- n essays: 100
- K active criteria: 10
- T comparisons: 2000

## Global parameters

- β̂ = +2.5987 (SE 0.1241) ; σ(β̂) = 0.9308
- NLL (regularized) = 7316.7187 ; NLL (unregularized) = 7315.6593
- L-BFGS-B iterations: 177 ; converged: True
- Convex sanity: |L_primary − L_sanity| = 1.453e-05

## Per-criterion table

| k | canonical_name | share | std(s_k) | std(s_k) SE | γ_k | γ_k SE | sign_corr |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | Organization and Coherence | 0.216 | 1.800 | 0.182 | +2.040 | 0.136 | +0.899 |
| 1 | Reasoning Development | 0.121 | 0.979 | 0.060 | -0.477 | 0.085 | +0.861 |
| 2 | Counterargument and Nuance | 0.111 | 1.859 | 0.111 | -0.528 | 0.105 | +0.877 |
| 3 | Thesis and Focus | 0.100 | 1.016 | 0.053 | -0.203 | 0.087 | +0.405 |
| 4 | Written Language Control | 0.086 | 1.565 | 0.081 | -1.858 | 0.114 | +0.829 |
| 5 | Analytical Depth | 0.044 | 0.952 | 0.059 | -2.090 | 0.121 | +0.738 |
| 6 | Conclusion Effectiveness | 0.036 | 0.811 | 0.063 | -2.215 | 0.137 | +0.630 |
| 7 | Formal Register | 0.036 | 1.797 | 0.104 | -4.115 | 0.206 | +0.809 |
| 8 | Rhetorical Effectiveness | 0.029 | 1.428 | 0.087 | -3.398 | 0.188 | +0.843 |
| 9 | Language Sophistication | 0.013 | 1.795 | 0.146 | -5.901 | 0.374 | +0.720 |

## Held-out validation

- n_train = 1600  n_test = 400
- BT-predictor winner accuracy = 0.9400
