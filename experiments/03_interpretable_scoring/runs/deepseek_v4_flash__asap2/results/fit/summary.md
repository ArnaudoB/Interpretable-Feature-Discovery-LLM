# Fit summary — `interpretable_scoring_deepseek_v4_flash__asap2`

- judge: `deepseek_v4_flash` (deepseek-v4-flash)
- dataset: `asap2_driverless`
- prompt variant: `comparative_verdict_qualities_asap2_srcarticle`
- n essays: 100
- K active criteria: 14
- T comparisons: 2000

## Global parameters

- β̂ = +3.3031 (SE 0.1742) ; σ(β̂) = 0.9645
- NLL (regularized) = 8274.6109 ; NLL (unregularized) = 8272.8236
- L-BFGS-B iterations: 497 ; converged: True
- Convex sanity: |L_primary − L_sanity| = 2.134e-07

## Per-criterion table

| k | canonical_name | share | std(s_k) | std(s_k) SE | γ_k | γ_k SE | sign_corr |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | Organization and Coherence | 0.216 | 1.559 | 0.156 | +2.184 | 0.143 | +0.861 |
| 1 | Evidence Use and Integration | 0.191 | 1.680 | 0.093 | +1.378 | 0.114 | +0.759 |
| 2 | Reasoning Development | 0.121 | 0.959 | 0.067 | -0.425 | 0.084 | +0.845 |
| 3 | Counterargument and Nuance | 0.111 | 1.858 | 0.115 | -0.447 | 0.105 | +0.862 |
| 4 | Thesis and Focus | 0.100 | 0.990 | 0.053 | -0.152 | 0.086 | +0.370 |
| 5 | Written Language Control | 0.086 | 1.559 | 0.084 | -1.810 | 0.116 | +0.819 |
| 6 | Analytical Depth | 0.044 | 0.982 | 0.062 | -2.045 | 0.122 | +0.695 |
| 7 | Conclusion Effectiveness | 0.036 | 0.798 | 0.057 | -2.111 | 0.133 | +0.554 |
| 8 | Formal Register | 0.036 | 1.862 | 0.115 | -4.148 | 0.224 | +0.793 |
| 9 | Rhetorical Effectiveness | 0.029 | 1.525 | 0.103 | -3.404 | 0.203 | +0.820 |
| 10 | Language Sophistication | 0.013 | 1.861 | 0.177 | -5.969 | 0.443 | +0.720 |
| 11 | Concrete Illustration | 0.009 | 1.657 | 0.158 | -5.831 | 0.445 | +0.731 |
| 12 | Source Fidelity | 0.006 | 1.493 | 0.238 | -5.577 | 0.606 | +0.574 |
| 13 | Constructive Proposal | 0.003 | 2.623 | 0.387 | -10.159 | 1.281 | +0.487 |

## Held-out validation

- n_train = 1600  n_test = 400
- BT-predictor winner accuracy = 0.9375
