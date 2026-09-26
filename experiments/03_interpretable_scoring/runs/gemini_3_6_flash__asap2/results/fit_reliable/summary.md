# Fit summary — `interpretable_scoring_gemini_3_6_flash__asap2`

- judge: `gemini_3_6_flash` (gemini-3.6-flash)
- dataset: `asap2_driverless`
- prompt variant: `comparative_verdict_qualities_asap2_srcarticle`
- n essays: 100
- K active criteria: 6
- T comparisons: 2000

## Global parameters

- β̂ = +0.8313 (SE 0.0828) ; σ(β̂) = 0.6966
- NLL (regularized) = 4304.1933 ; NLL (unregularized) = 4303.4827
- L-BFGS-B iterations: 133 ; converged: True
- Convex sanity: |L_primary − L_sanity| = 2.774e-08

## Per-criterion table

| k | canonical_name | share | std(s_k) | std(s_k) SE | γ_k | γ_k SE | sign_corr |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | Source Evidence Integration | 0.172 | 2.081 | 0.098 | -1.223 | 0.118 | +0.896 |
| 1 | Reasoning and Elaboration | 0.163 | 1.015 | 0.060 | -0.544 | 0.090 | +0.849 |
| 2 | Conventions and Mechanics | 0.108 | 2.163 | 0.099 | -3.135 | 0.147 | +0.869 |
| 3 | Thesis and Stance | 0.037 | 1.108 | 0.073 | -3.215 | 0.168 | +0.644 |
| 4 | Cohesion and Transitions | 0.028 | 1.161 | 0.096 | -3.694 | 0.234 | +0.700 |
| 5 | Focus and Task Relevance | 0.019 | 1.263 | 0.111 | -4.428 | 0.280 | +0.452 |

## Held-out validation

- n_train = 1600  n_test = 400
- BT-predictor winner accuracy = 0.9100
