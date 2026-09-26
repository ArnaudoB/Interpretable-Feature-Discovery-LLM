# Fit summary — `interpretable_scoring_claude_haiku_4_5__asap2`

- judge: `claude_haiku_4_5` (claude-haiku-4-5)
- dataset: `asap2_driverless`
- prompt variant: `comparative_verdict_qualities_asap2_srcarticle`
- n essays: 100
- K active criteria: 18
- T comparisons: 2000

## Global parameters

- β̂ = +0.5289 (SE 0.1147) ; σ(β̂) = 0.6292
- NLL (regularized) = 10507.2318 ; NLL (unregularized) = 10504.7120
- L-BFGS-B iterations: 428 ; converged: True
- Convex sanity: |L_primary − L_sanity| = 3.092e-07

## Per-criterion table

| k | canonical_name | share | std(s_k) | std(s_k) SE | γ_k | γ_k SE | sign_corr |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | Organization and Structural Coherence | 0.177 | 2.180 | 0.154 | +0.003 | 0.112 | +0.937 |
| 1 | Evidence Integration and Textual Support | 0.163 | 1.377 | 0.083 | +0.266 | 0.098 | +0.878 |
| 2 | Logical Reasoning and Explanation | 0.120 | 0.736 | 0.052 | -0.275 | 0.086 | +0.852 |
| 3 | Counterargument Engagement | 0.106 | 1.380 | 0.082 | -0.750 | 0.102 | +0.905 |
| 4 | Mechanics, Grammar, and Sentence Control | 0.087 | 1.722 | 0.088 | -2.144 | 0.129 | +0.859 |
| 5 | Argument Focus and Thesis | 0.084 | 1.034 | 0.055 | -0.793 | 0.094 | +0.496 |
| 6 | Nuance, Complexity, and Broader Implications | 0.060 | 1.149 | 0.061 | -1.407 | 0.115 | +0.776 |
| 7 | Argument Development and Breadth | 0.054 | 0.868 | 0.058 | -1.844 | 0.116 | +0.821 |
| 8 | Conclusion Effectiveness and Closure | 0.041 | 1.114 | 0.079 | -2.847 | 0.151 | +0.855 |
| 9 | Concrete Illustration and Specificity | 0.030 | 1.511 | 0.090 | -2.684 | 0.174 | +0.767 |
| 10 | Source Understanding and Engagement | 0.022 | 1.184 | 0.087 | -2.358 | 0.169 | +0.439 |
| 11 | Contextual Framing and Introduction | 0.013 | 1.325 | 0.129 | -4.588 | 0.319 | +0.746 |
| 12 | Style, Register, and Voice | 0.011 | 1.771 | 0.166 | -5.952 | 0.412 | +0.732 |
| 13 | Rhetorical Effectiveness and Audience Engagement | 0.008 | 2.108 | 0.278 | -6.680 | 0.775 | +0.699 |
| 14 | Task Fulfillment and Completeness | 0.007 | 2.089 | 0.369 | -6.940 | 0.928 | +0.656 |
| 15 | Solution Orientation | 0.006 | 2.856 | 0.333 | -8.462 | 1.048 | +0.531 |
| 16 | Comparative and Analogical Reasoning | 0.005 | 2.056 | 0.272 | -7.656 | 0.812 | +0.607 |
| 17 | Accurate Source Representation | 0.005 | 2.046 | 0.373 | -6.817 | 1.040 | +0.645 |

## Held-out validation

- n_train = 1600  n_test = 400
- BT-predictor winner accuracy = 0.9425
