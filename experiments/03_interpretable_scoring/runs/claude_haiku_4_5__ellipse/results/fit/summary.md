# Fit summary — `interpretable_scoring_claude_haiku_4_5__ellipse`

- judge: `claude_haiku_4_5` (claude-haiku-4-5)
- dataset: `ellipse`
- prompt variant: `comparative_verdict_qualities`
- n essays: 100
- K active criteria: 26
- T comparisons: 2000

## Global parameters

- β̂ = -1.1892 (SE 0.2144) ; σ(β̂) = 0.2334
- NLL (regularized) = 11276.3634 ; NLL (unregularized) = 11264.9284
- L-BFGS-B iterations: 2496 ; converged: True
- Convex sanity: |L_primary − L_sanity| = 2.528e-06

## Per-criterion table

| k | canonical_name | share | std(s_k) | std(s_k) SE | γ_k | γ_k SE | sign_corr |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | Organization and Coherence | 0.181 | 3.018 | 0.227 | +2.591 | 0.235 | +0.775 |
| 1 | Sentence Variety | 0.130 | 1.627 | 0.087 | +0.760 | 0.109 | +0.538 |
| 2 | Support and Elaboration | 0.123 | 1.204 | 0.072 | -0.057 | 0.095 | +0.868 |
| 3 | Grammar Accuracy | 0.119 | 1.031 | 0.062 | +0.208 | 0.090 | +0.607 |
| 4 | Vocabulary Control | 0.080 | 1.716 | 0.109 | -1.924 | 0.116 | +0.930 |
| 5 | Thesis Focus | 0.059 | 0.742 | 0.046 | -0.858 | 0.094 | +0.545 |
| 6 | Spelling | 0.055 | 1.313 | 0.070 | -1.612 | 0.117 | +0.393 |
| 7 | Counterargument and Nuance | 0.048 | 1.954 | 0.105 | -2.150 | 0.146 | +0.804 |
| 8 | Sentence Construction | 0.046 | 1.650 | 0.075 | -1.865 | 0.125 | +0.747 |
| 9 | Mechanics | 0.030 | 1.278 | 0.097 | -3.178 | 0.187 | +0.748 |
| 10 | Reasoning and Explanation | 0.028 | 0.854 | 0.062 | -2.501 | 0.131 | +0.795 |
| 11 | Technical Accuracy | 0.019 | 1.139 | 0.088 | -3.261 | 0.195 | +0.654 |
| 12 | Argument Development | 0.017 | 0.745 | 0.064 | -2.853 | 0.167 | +0.670 |
| 13 | Comprehensibility | 0.013 | 1.668 | 0.103 | -3.939 | 0.262 | +0.717 |
| 14 | Rhetorical Sophistication | 0.012 | 1.133 | 0.097 | -3.753 | 0.233 | +0.697 |
| 15 | Register | 0.010 | 1.745 | 0.169 | -5.335 | 0.467 | +0.628 |
| 16 | Narrative Development | 0.006 | 3.020 | 0.291 | -6.293 | 0.752 | +0.542 |
| 17 | Conceptual Sophistication | 0.005 | 2.855 | 0.234 | -5.991 | 0.649 | +0.489 |
| 18 | Source Integration | 0.004 | 3.508 | 0.482 | -12.064 | 1.597 | +0.636 |
| 19 | Audience Awareness | 0.003 | 2.784 | 0.548 | -10.421 | 1.905 | +0.661 |
| 20 | Genre Awareness | 0.003 | 2.172 | 0.182 | -7.515 | 0.763 | +0.549 |
| 21 | Conclusion | 0.003 | 2.762 | 0.606 | -9.980 | 1.884 | +0.661 |
| 22 | Completeness | 0.002 | 8.508 | 0.777 | -37.589 | 3.271 | +0.449 |
| 23 | Language Consistency | 0.002 | 7.526 | 0.968 | -35.901 | 6.935 | +0.396 |
| 24 | Voice | 0.002 | 2.000 | 0.282 | -8.445 | 1.031 | +0.644 |
| 25 | Conciseness | 0.001 | 2.940 | 0.554 | -10.755 | 3.078 | +0.483 |

## Held-out validation

- n_train = 1600  n_test = 400
- BT-predictor winner accuracy = 0.9400
