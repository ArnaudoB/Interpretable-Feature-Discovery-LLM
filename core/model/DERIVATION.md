# Derivation

The direction/citation gated Bradley–Terry model used for the comparative arm
(`gated.py`; Section `sec:method` and Appendix `app:model` of the paper).
Gradients and the expected Hessian are analytic (no autodiff). The finite-difference
gradient check (`python -m core.model.gated --gradcheck`) guards the algebra.

## Model

Parameters `s ∈ R^{n×K}`, `γ ∈ R^K`, `β ∈ R`. For pair `t` (essay `a_t` in slot A,
`b_t` in slot B) and criterion `k`, let `Δ = s[a_t,k] − s[b_t,k]` and
`X_{t,k} ∈ {−1, 0, +1}` (+1 = A favored, −1 = B favored, 0 = not cited).

- **Citation (gate):** `z_cite = log(2 cosh Δ) + γ_k`, `p_cite = σ(z_cite)`.
  β does **not** enter. The gate is driven by the *magnitude* of Δ.
- **Direction | cited:** `z_dir = Δ + β`, `p_dir = σ(z_dir) = P(X=+1 | cited)`.
- Per-obs log-likelihood
  `L = 1[X≠0]·log p_cite + 1[X=0]·log(1−p_cite) + 1[X=+1]·log p_dir + 1[X=−1]·log(1−p_dir)`.
- `NLL = −Σ_{t,k} L + ½·ridge·‖s‖²` (ridge on s only).

## Residuals and weights

`cited = 1[X≠0]`, `pos = 1[X=+1]`. `r_cite = cited − p_cite`,
`d_dir = pos − cited·p_dir`. `w_cite = p_cite(1−p_cite)`, `w_dir = p_dir(1−p_dir)`.

## Gradient (chain through Δ)

`∂z_cite/∂Δ = tanh Δ`, `∂z_dir/∂Δ = ∂z_dir/∂β = 1`, `∂z_cite/∂γ_k = 1`.

- `dL/dΔ = r_cite·tanh Δ + d_dir`
- `dL/ds[i,k] = Σ_{t:a_t=i} dL/dΔ − Σ_{t:b_t=i} dL/dΔ` (then `+ ridge·s` in the NLL)
- `dL/dγ_k = Σ_t r_cite[t,k]`
- **`dL/dβ = Σ_{t,k} d_dir[t,k]`** — the citation term contributes nothing because β is
  absent from the gate. (Had β entered through `Δ+β` in *both* logits, the citation
  residual would also reach it.)

## Expected (Fisher) Hessian

Drop the `r·(second derivative)` middle terms (mean zero) → PSD weights:

- `h_DD = w_cite·tanh²Δ + cited·w_dir`  (score–score)
- `h_Dg = w_cite·tanh Δ`  (score–γ)
- `h_gg = w_cite`  (γ–γ)
- **s–β coupling = `cited·w_dir`** (scatter by essay), **not** `h_DD`
- **γ–β coupling = 0** (β and γ_k share no logit)
- **β–β = `Σ cited·w_dir`**, **not** `Σ h_DD`

The (s, γ) block is block-diagonal by criterion; β borders only the s-block.

## Initialization (symmetric point Δ=0)

`log(2 cosh 0) = log 2`, `z_dir|_{Δ=0} = β`:

- `s = 0`
- `γ_k = logit(p_cite_k) − log 2`
- `β = logit(p_{A|cited})`

(`z_dir` has unit slope in β, and the γ correction is exactly `log 2` since β does not
enter the gate).

## Convexity

The cited-citation term `−log σ(log2cosh Δ + γ)` has observed second derivative
`w_cite·tanh²Δ − r_cite·sech²Δ`, which is indefinite near Δ=0 (where `tanh²→0` but
`sech²→1`). So the finite-sample NLL is **mildly non-convex**. Mitigations: ridge>0 strictly convexifies the s-block; the informative init starts
near the right citation/direction rates; the mean-zero s projection removes the constant
null space. The *expected* Hessian drops the indefinite `−r_cite·sech²` term and is PSD by
construction, so it remains the correct object for κ (identification) and the sandwich
noise (reliability).
