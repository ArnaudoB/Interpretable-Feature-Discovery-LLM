# Fit summary — `interpretable_scoring_claude_haiku_4_5__ellipse`

- judge: `claude_haiku_4_5` (claude-haiku-4-5)
- dataset: `ellipse`
- prompt variant: `comparative_verdict_qualities`
- n essays: 100
- K active criteria: 9
- T comparisons: 2000

## Global parameters

- β̂ = -0.4566 (SE 0.0877) ; σ(β̂) = 0.3878
- NLL (regularized) = 7491.7506 ; NLL (unregularized) = 7490.9360
- L-BFGS-B iterations: 145 ; converged: True
- Convex sanity: |L_primary − L_sanity| = 2.570e-05

## Per-criterion table

| k | canonical_name | share | std(s_k) | std(s_k) SE | γ_k | γ_k SE | sign_corr |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | Sentence Variety | 0.130 | 1.439 | 0.077 | +0.510 | 0.093 | +0.645 |
| 1 | Support and Elaboration | 0.123 | 1.349 | 0.092 | -0.236 | 0.097 | +0.891 |
| 2 | Grammar Accuracy | 0.119 | 1.085 | 0.064 | +0.070 | 0.090 | +0.692 |
| 3 | Vocabulary Control | 0.080 | 1.870 | 0.112 | -2.160 | 0.126 | +0.930 |
| 4 | Thesis Focus | 0.059 | 0.784 | 0.047 | -1.024 | 0.097 | +0.635 |
| 5 | Counterargument and Nuance | 0.048 | 1.628 | 0.078 | -2.243 | 0.134 | +0.906 |
| 6 | Mechanics | 0.030 | 1.473 | 0.109 | -3.604 | 0.211 | +0.796 |
| 7 | Reasoning and Explanation | 0.028 | 0.947 | 0.067 | -2.770 | 0.142 | +0.813 |
| 8 | Rhetorical Sophistication | 0.012 | 1.180 | 0.098 | -4.155 | 0.252 | +0.758 |

## Held-out validation

- n_train = 1600  n_test = 400
- BT-predictor winner accuracy = 0.9375
