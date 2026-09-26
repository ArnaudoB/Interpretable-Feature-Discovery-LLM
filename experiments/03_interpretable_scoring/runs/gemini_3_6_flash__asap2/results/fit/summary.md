# Fit summary — `interpretable_scoring_gemini_3_6_flash__asap2`

- judge: `gemini_3_6_flash` (gemini-3.6-flash)
- dataset: `asap2_driverless`
- prompt variant: `comparative_verdict_qualities_asap2_srcarticle`
- n essays: 100
- K active criteria: 17
- T comparisons: 2000

## Global parameters

- β̂ = +1.5653 (SE 0.1630) ; σ(β̂) = 0.8271
- NLL (regularized) = 7256.1866 ; NLL (unregularized) = 7251.7022
- L-BFGS-B iterations: 1268 ; converged: True
- Convex sanity: |L_primary − L_sanity| = 3.044e-07

## Per-criterion table

| k | canonical_name | share | std(s_k) | std(s_k) SE | γ_k | γ_k SE | sign_corr |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | Structural Organization | 0.259 | 3.373 | 0.275 | +0.656 | 0.110 | +0.623 |
| 1 | Source Evidence Integration | 0.172 | 2.067 | 0.106 | -1.082 | 0.121 | +0.887 |
| 2 | Reasoning and Elaboration | 0.163 | 0.953 | 0.057 | -0.418 | 0.089 | +0.813 |
| 3 | Conventions and Mechanics | 0.108 | 2.097 | 0.102 | -2.969 | 0.149 | +0.854 |
| 4 | Counterargument and Nuance | 0.096 | 1.795 | 0.094 | -1.578 | 0.130 | +0.785 |
| 5 | Thesis and Stance | 0.037 | 1.246 | 0.104 | -3.144 | 0.186 | +0.608 |
| 6 | Cohesion and Transitions | 0.028 | 0.983 | 0.073 | -3.122 | 0.186 | +0.577 |
| 7 | Sentence Fluency and Syntax | 0.020 | 1.383 | 0.109 | -4.150 | 0.275 | +0.709 |
| 8 | Academic Tone and Register | 0.019 | 2.707 | 0.268 | -7.606 | 0.659 | +0.724 |
| 9 | Paragraph Development | 0.019 | 0.950 | 0.101 | -3.102 | 0.209 | +0.531 |
| 10 | Focus and Task Relevance | 0.019 | 1.467 | 0.103 | -4.163 | 0.313 | +0.487 |
| 11 | Rhetorical Engagement | 0.015 | 1.789 | 0.172 | -4.424 | 0.414 | +0.565 |
| 12 | Introduction and Framing | 0.015 | 2.064 | 0.216 | -5.693 | 0.588 | +0.611 |
| 13 | Originality of Thought | 0.012 | 2.898 | 0.259 | -6.817 | 0.673 | +0.641 |
| 14 | Vocabulary and Style | 0.012 | 2.956 | 0.476 | -9.200 | 1.279 | +0.653 |
| 15 | Conclusion | 0.004 | 1.756 | 0.304 | -6.720 | 0.967 | +0.614 |
| 16 | Source Accuracy | 0.001 | 4.797 | 0.804 | -17.079 | 2.350 | +0.373 |

## Held-out validation

- n_train = 1600  n_test = 400
- BT-predictor winner accuracy = 0.9425
