# Fit summary — `interpretable_scoring_claude_haiku_4_5__asap2`

- judge: `claude_haiku_4_5` (claude-haiku-4-5)
- dataset: `asap2_driverless`
- prompt variant: `comparative_verdict_qualities_asap2_srcarticle`
- n essays: 100
- K active criteria: 16
- T comparisons: 2000

## Global parameters

- β̂ = +0.5360 (SE 0.1146) ; σ(β̂) = 0.6309
- NLL (regularized) = 10360.8714 ; NLL (unregularized) = 10359.0642
- L-BFGS-B iterations: 317 ; converged: True
- Convex sanity: |L_primary − L_sanity| = 8.913e-07

## Per-criterion table

| k | canonical_name | share | std(s_k) | std(s_k) SE | γ_k | γ_k SE | sign_corr |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | Organization and Structural Coherence | 0.177 | 2.151 | 0.156 | -0.008 | 0.110 | +0.937 |
| 1 | Evidence Integration and Textual Support | 0.163 | 1.370 | 0.082 | +0.257 | 0.098 | +0.880 |
| 2 | Logical Reasoning and Explanation | 0.120 | 0.728 | 0.051 | -0.280 | 0.086 | +0.856 |
| 3 | Counterargument Engagement | 0.106 | 1.396 | 0.088 | -0.759 | 0.102 | +0.904 |
| 4 | Mechanics, Grammar, and Sentence Control | 0.087 | 1.717 | 0.087 | -2.145 | 0.128 | +0.862 |
| 5 | Argument Focus and Thesis | 0.084 | 1.024 | 0.054 | -0.797 | 0.094 | +0.500 |
| 6 | Nuance, Complexity, and Broader Implications | 0.060 | 1.177 | 0.063 | -1.431 | 0.116 | +0.783 |
| 7 | Argument Development and Breadth | 0.054 | 0.871 | 0.058 | -1.852 | 0.116 | +0.820 |
| 8 | Conclusion Effectiveness and Closure | 0.041 | 1.098 | 0.079 | -2.832 | 0.149 | +0.855 |
| 9 | Concrete Illustration and Specificity | 0.030 | 1.490 | 0.095 | -2.720 | 0.179 | +0.788 |
| 10 | Source Understanding and Engagement | 0.022 | 1.172 | 0.089 | -2.409 | 0.174 | +0.451 |
| 11 | Contextual Framing and Introduction | 0.013 | 1.307 | 0.127 | -4.581 | 0.309 | +0.743 |
| 12 | Style, Register, and Voice | 0.011 | 1.758 | 0.160 | -5.925 | 0.396 | +0.733 |
| 13 | Rhetorical Effectiveness and Audience Engagement | 0.008 | 1.895 | 0.203 | -6.187 | 0.573 | +0.693 |
| 14 | Task Fulfillment and Completeness | 0.007 | 1.837 | 0.259 | -6.431 | 0.647 | +0.618 |
| 15 | Accurate Source Representation | 0.005 | 2.111 | 0.293 | -7.019 | 0.864 | +0.630 |

## Held-out validation

- n_train = 1600  n_test = 400
- BT-predictor winner accuracy = 0.9350
