# Holistic-verdict model: joint MLE — derivation

This document records the math behind `core.model.holistic`. It is the paper-appendix
source of truth; the docstrings of the modules paraphrase but do not repeat
the full derivation.

---

## 1. Notation

| Symbol | Type | Meaning |
|---|---|---|
| $n$ | int | number of items (essays, summaries, …) |
| $K$ | int | number of latent quality criteria |
| $T$ | int | number of pairwise comparisons |
| $a_t, b_t \in \{0, \dots, n-1\}$ | int | slot-A and slot-B item indices for comparison $t$ |
| $w_t \in \{0, 1\}$ | int | $w_t = 1$ iff slot-A won |
| $\eta_t = 2w_t - 1 \in \{-1, +1\}$ | int | winner sign (slot-A winner $\Rightarrow +1$) |
| $r_{t,k} \in \{0, 1\}$ | int | mention indicator for criterion $k$ in comparison $t$ |
| $s \in \mathbb{R}^{n \times K}$ | param | per-item, per-criterion latent score |
| $\gamma \in \mathbb{R}^K$ | param (optional) | per-criterion mention intercept |
| $\beta \in \mathbb{R}$ | param | global position bias on the BT predictor |
| $\theta = (\mathrm{vec}(s), \gamma, \beta)$ | param | flat parameter vector, dimension $P$ |
| $P$ | int | $P = nK + K + 1$ with $\gamma$; $P = nK + 1$ without |
| $A \in \mathbb{R}^{T \times n}$ | data | $A_{t,i} = \mathbb{1}[a_t = i] - \mathbb{1}[b_t = i]$ |
| $\lambda$ | scalar | ridge coefficient (default $10^{-3}$) |
| $\sigma(z)$ | scalar | logistic sigmoid, $\sigma(z) = 1/(1+e^{-z})$ |

Flat-vector layout (row-major C order on $s$):
$$
\theta = \bigl[\, s_{0,0},\, s_{0,1},\, \dots,\, s_{0,K-1},\, s_{1,0},\, \dots,\, s_{n-1,K-1},\, \gamma_0,\, \dots,\, \gamma_{K-1},\, \beta \,\bigr].
$$
The $s_{i,k}$ flat index is $i \cdot K + k$.

---

## 2. Likelihood

Define the two predictors per comparison $t$:
$$
\begin{aligned}
z^{\text{BT}}_t &= \sum_{k=0}^{K-1} (s_{a_t, k} - s_{b_t, k}) + \beta
                  = A_t s \mathbf{1}_K + \beta, \\
z^{\text{M}}_{t,k} &= \eta_t (s_{a_t, k} - s_{b_t, k}) + \gamma_k.
\end{aligned}
$$

The data generating process is two independent Bernoulli channels per
comparison:
$$
\Pr(w_t = 1 \mid s, \beta) = \sigma\bigl(z^{\text{BT}}_t\bigr),
\qquad
\Pr(r_{t,k} = 1 \mid s, \gamma, w_t) = \sigma\bigl(z^{\text{M}}_{t,k}\bigr).
$$

Let $p_t := \sigma(z^{\text{BT}}_t)$ and $q_{t,k} := \sigma(z^{\text{M}}_{t,k})$. The
ridge-regularized negative log-likelihood is
$$
\mathcal{L}(\theta) \;=\;
\underbrace{-\sum_t \bigl[\, w_t \log p_t + (1 - w_t) \log(1 - p_t) \,\bigr]}_{\mathcal{L}^{\text{BT}}}
\;+\;
\underbrace{-\sum_t \sum_k \bigl[\, r_{t,k} \log q_{t,k} + (1 - r_{t,k}) \log(1 - q_{t,k}) \,\bigr]}_{\mathcal{L}^{\text{M}}}
\;+\;
\tfrac{\lambda}{2} \|s\|_F^2.
$$

Both predictors are **affine** in $\theta$, and the binary cross-entropy
$-\log\sigma(\cdot)$ is convex. A sum of convex functions of affine maps of
$\theta$ is convex, so $\mathcal{L}(\theta)$ is convex on $\mathbb{R}^P$. The
ridge $\tfrac{\lambda}{2}\|s\|_F^2$ is strictly convex on the $s$ block.

---

## 3. Identifiability

### 3.1 Per-column shift symmetry

Consider the transformation
$$
s_{·,k} \mapsto s_{·,k} + \alpha_k \mathbf{1}_n, \qquad k = 0, \dots, K-1,
$$
for arbitrary $\alpha \in \mathbb{R}^K$. Both predictors are invariant:

- **Mention predictor.** $\eta_t \bigl[(s_{a,k} + \alpha_k) - (s_{b,k} + \alpha_k)\bigr] + \gamma_k = \eta_t(s_{a,k} - s_{b,k}) + \gamma_k$. The $\alpha_k$ cancels because the predictor uses a *difference* in $s$, and $\gamma$ is untouched.
- **BT predictor.** $\sum_k \bigl[(s_{a,k} + \alpha_k) - (s_{b,k} + \alpha_k)\bigr] + \beta = \sum_k(s_{a,k} - s_{b,k}) + \beta$. Same cancellation, column by column.

So the unregularized NLL has a **$K$-dimensional flat null subspace**: one
direction per column of $s$ (each direction is "shift column $k$ by the all-ones
vector"). The model is unidentified without further constraint.

> Aside: an offhand claim that "the BT term is sensitive to global shifts in $s$"
> is *false* for per-column shifts. It is true that adding the same vector
> $v \in \mathbb{R}^K$ to *every row* of $s$ (i.e. shifting items uniformly) is
> the same thing as $K$ different $\alpha_k$ values applied independently, so
> there is no additional symmetry beyond the per-column-shift one.

### 3.2 Ridge as identification

The ridge term $\tfrac{\lambda}{2}\|s\|_F^2$ is strictly convex on the
null subspace (a non-zero shift $\alpha$ strictly increases $\|s\|_F^2$).
This breaks all $K$ null directions and selects the **zero-column-mean
representative**:

Stationarity of $\nabla_{s_{i,k}}(\mathcal{L} + \tfrac{\lambda}{2}\|s\|_F^2)$
summed over $i$: the data gradient summed over $i$ for fixed $k$ is zero
(for any pair $(a, b)$, the gradient at index $a$ is $+\xi$ and at $b$ is
$-\xi$, so the sum cancels). The ridge gradient summed over $i$ is
$\lambda n \bar{s}_k$. Setting the total to zero forces
$\bar{s}_k = \tfrac{1}{n}\sum_i s_{i,k} = 0$ at the optimum.

Consequence: at convergence the column means of $\hat s$ are zero to
floating-point precision, and the post-hoc centering applied by `fit()`
is empirically a no-op (it just scrubs $10^{-12}$-scale residuals).

### 3.3 Sign convention

The prompt design ("list specific qualities the WINNER has MORE of") fixes
the sign of $\gamma$: a positive $\gamma_k$ means the judge frequently cites
criterion $k$ even at the average-quality item. Within this convention, the
sign of $\hat s_{·,k}$ is fully determined by the data: higher $\hat s_{i,k}$
should correspond to items that are more often the *winner* in comparisons
where criterion $k$ is cited.

We verify this empirically with `sign_corrs`:
$$
\text{sign\_corr}_k = \text{Pearson}\!\left(\hat s_{·,k},\;
  \text{won\_cited}_{·,k}\right),
\quad \text{won\_cited}_{i,k} = \bigl|\{t : \text{winner}(t) = i \,\wedge\, r_{t,k} = 1\}\bigr|.
$$
Sensible fits have $\text{sign\_corr}_k > 0$ for every $k$; the fitter
warns otherwise. (We use Pearson here, not Spearman — Pearson captures
linear alignment, which is what the model's score $\hat s$ posits.)

---

## 4. Gradient

Write the per-pair residuals
$$
u_t = p_t - w_t, \qquad v_{t,k} = q_{t,k} - r_{t,k}.
$$

**With respect to $s_{i,k}$.** The BT contribution depends only on whether
$i = a_t$ or $i = b_t$, and is the *same value for every column* $k$ because
the BT predictor uses the row sum of $s$:
$$
\frac{\partial \mathcal{L}^{\text{BT}}}{\partial s_{i,k}}
  = \sum_{t : a_t = i} u_t \;-\; \sum_{t : b_t = i} u_t.
$$
The mention contribution is per-$(i, k)$ and uses the winner sign $\eta_t$:
$$
\frac{\partial \mathcal{L}^{\text{M}}}{\partial s_{i,k}}
  = \sum_{t : a_t = i} \eta_t \, v_{t,k} \;-\; \sum_{t : b_t = i} \eta_t \, v_{t,k}.
$$
Plus the ridge term: $\lambda s_{i,k}$.

**With respect to $\gamma_k$.** Only the mention term contributes:
$$
\frac{\partial \mathcal{L}}{\partial \gamma_k} = \sum_t v_{t,k}.
$$

**With respect to $\beta$.** Only the BT term:
$$
\frac{\partial \mathcal{L}}{\partial \beta} = \sum_t u_t.
$$

Implementation uses `np.add.at` scatter-adds to vectorize the index unions.

---

## 5. Hessian

Use $s^{\text{BT}}_t := p_t(1 - p_t)$ and $r^{\text{M}}_{t,k} := q_{t,k}(1 - q_{t,k})$
for the Bernoulli curvature factors. Define $A \in \mathbb{R}^{T \times n}$
as in §1. Note $\eta_t^2 = 1$, so the mention block does not retain the $\eta$
sign at second order.

### 5.1 $s$ block — $H_{ss}$

The BT contribution to $\partial^2 \mathcal{L}^{\text{BT}} / \partial s_{i,k} \, \partial s_{i', k'}$ is the same for every $(k, k')$ pair because the BT predictor uses
the row sum:
$$
(H_{ss}^{\text{BT}})_{(i, k), (i', k')}
  = \sum_t s^{\text{BT}}_t \, A_{t, i} A_{t, i'}
  = \bigl(A^T \mathrm{diag}(s^{\text{BT}}) A\bigr)_{i, i'}.
$$
In Kronecker form, $H_{ss}^{\text{BT}} = (A^T \mathrm{diag}(s^{\text{BT}}) A) \otimes \mathbf{1}_{K \times K}$.

The mention contribution decouples across $k$:
$$
(H_{ss}^{\text{M}})_{(i, k), (i', k')}
  = \delta_{k = k'} \sum_t r^{\text{M}}_{t,k} \, \eta_t^2 \, A_{t,i} A_{t, i'}
  = \delta_{k = k'} \bigl(A^T \mathrm{diag}(r^{\text{M}}_{·,k}) A\bigr)_{i, i'},
$$
so it appears only on the diagonal-in-$k$ block (an $(n \times n)$ matrix
added to each $k = k'$ slice).

Ridge adds $\lambda \cdot I_{nK}$.

### 5.2 $s, \gamma$ cross block — $H_{s\gamma}$

Only the mention term contributes, and only at $k = k'$:
$$
\frac{\partial^2 \mathcal{L}}{\partial s_{i, k} \, \partial \gamma_{k'}}
  = \delta_{k = k'} \sum_t \eta_t \, r^{\text{M}}_{t, k} \, A_{t, i}.
$$
Diagonal in $(k, k')$.

### 5.3 $s, \beta$ cross block — $H_{s\beta}$

Only the BT term, and the value is the *same for every $k$*:
$$
\frac{\partial^2 \mathcal{L}}{\partial s_{i, k} \, \partial \beta}
  = \sum_t s^{\text{BT}}_t \, A_{t, i}.
$$

### 5.4 $\gamma, \gamma$ block — $H_{\gamma\gamma}$

Diagonal:
$$
\frac{\partial^2 \mathcal{L}}{\partial \gamma_k^2} = \sum_t r^{\text{M}}_{t, k}, \qquad
\frac{\partial^2 \mathcal{L}}{\partial \gamma_k \, \partial \gamma_{k'}} = 0 \quad (k \ne k').
$$

### 5.5 $\gamma, \beta$ cross block — $H_{\gamma\beta} = 0$

The two parameters live in disjoint terms ($\gamma$ in mention, $\beta$ in BT).

### 5.6 $\beta, \beta$ block — $H_{\beta\beta}$

$$
\frac{\partial^2 \mathcal{L}}{\partial \beta^2} = \sum_t s^{\text{BT}}_t.
$$

### 5.7 Assembly

For $n = 100$, $K = 7$ (typical operating point), $P = 708$; the dense
Hessian is $4$ MB. We assemble densely for simplicity. The structure
above suggests several efficient block-wise solves (Schur complement on
the $s$ block, e.g.) if memory becomes binding at larger $K$.

---

## 6. Sandwich confidence intervals

### 6.1 M-estimator CLT

For an M-estimator $\hat\theta = \arg\min_\theta \sum_t \rho_t(\theta)$ with smooth
$\rho_t$, the standard sandwich result is
$$
\sqrt{T}\,(\hat\theta - \theta^\star) \;\xrightarrow{d}\; \mathcal{N}\bigl(0, \;A^{-1} B A^{-1}\bigr),
$$
where $A = \mathbb{E}[\nabla^2 \rho_t(\theta^\star)]$ and $B = \mathbb{E}[\nabla \rho_t(\theta^\star) \nabla \rho_t(\theta^\star)^T]$.

Under correct specification, the information equality $A = B$ holds and
$\widehat{\mathrm{Cov}} = A^{-1}/T$ collapses to the usual ML CI. Under
misspecification (model error, heteroscedasticity), $A \ne B$ and the
sandwich is the asymptotically correct covariance.

### 6.2 Empirical estimators

Absorb factors of $T$ into the sums:
$$
\hat A = \sum_t \nabla^2 \rho_t(\hat\theta) + \lambda I_s = \hat H, \qquad
\hat B = \sum_t \nabla \rho_t(\hat\theta) \, \nabla \rho_t(\hat\theta)^T = \hat J.
$$
The covariance estimator is
$$
\widehat{\mathrm{Cov}}(\hat\theta) = (\hat H + \epsilon I)^{-1} \, \hat J \, (\hat H + \epsilon I)^{-1},
$$
where $\epsilon = 10^{-10}$ is a tiny numerical jitter added for invertibility
(not a second copy of the ridge — the ridge is already in $\hat H$).

### 6.3 Per-pair scores

The per-pair NLL is
$$
\rho_t(\theta) \;=\; -\bigl[w_t \log p_t + (1 - w_t) \log(1 - p_t)\bigr]
                  - \sum_k \bigl[r_{t,k} \log q_{t,k} + (1 - r_{t,k}) \log(1 - q_{t,k})\bigr].
$$
The score $\nabla \rho_t(\hat\theta)$ is sparse: nonzero entries only at
$s_{a_t, ·}$ (length $K$), $s_{b_t, ·}$ (length $K$), $\gamma$ (length $K$,
when included), and $\beta$ (length 1). Concretely:

- $(\partial \rho_t / \partial s_{a_t, k}) = u_t + \eta_t v_{t,k}$
- $(\partial \rho_t / \partial s_{b_t, k}) = -u_t - \eta_t v_{t,k}$
- $(\partial \rho_t / \partial \gamma_k) = v_{t,k}$
- $(\partial \rho_t / \partial \beta) = u_t$

We accumulate $\hat J$ by sparse rank-1 outer-product updates over the
$\le 3K + 1$ support indices per pair, avoiding a dense $(T, P)$ score
matrix. For $T = 3000$, $K = 7$, $P = 708$, the dense matrix would be
$\sim 17\,000$ entries per row and the storage cost is mild — but the
sparse path scales to larger $K$ better.

### 6.4 Calibration

The test `tests/test_holistic_model.py::test_sandwich_ci_calibration` (marked
`@slow`) generates 40 replications at $n = 20, K = 3, T = 500$ with fixed
ground truth, fits each, and compares the mean predicted SE on $\beta$
to the empirical standard deviation across reps. The acceptance band is
$0.8 < \text{predicted}/\text{empirical} < 1.2$.

---

## 7. Post-hoc normalization

We column-center $\hat s$ after the L-BFGS optimum:
$$
\hat s_{·, k} \;\leftarrow\; \hat s_{·, k} - \bar{\hat s}_k \mathbf{1}_n, \quad \text{where} \quad \bar{\hat s}_k = \tfrac{1}{n}\sum_i \hat s_{i, k}.
$$
$\gamma$ is *not* adjusted. Justification: as shown in §3.1, both predictors
are invariant to per-column shifts of $s$ (the mention term cancels because of
the difference structure; the BT term cancels column by column). So
column-centering changes neither $z^{\text{BT}}_t$ nor $z^{\text{M}}_{t,k}$ for any $t, k$,
hence neither $\mathcal{L}^{\text{BT}}$ nor $\mathcal{L}^{\text{M}}$, hence the NLL is exactly
preserved (modulo floating-point).

Practical reading: under the column-centered convention, an "average-quality"
item $i^\star$ has $\hat s_{i^\star, k} \approx 0$ for all $k$, so its mention
predictor is $z^{\text{M}}_{t,k} \approx \pm\hat\gamma_k$. The intercept $\hat\gamma_k$
is **the log-odds of citing criterion $k$ in a comparison between two
average-quality items** (up to the sign of the comparison outcome). This
matches the "what does the judge talk about at baseline" interpretation
required for cross-judge analyses.

The centering is also empirically a no-op modulo floating-point under the
ridge — see §3.2. It exists primarily as a defensive scrub.

---

## 8. Comparison to a multiplicative parametrization

An alternative is the
multiplicative parametrization $M = c \odot s$ with $c \in \mathbb{R}^K$ and $s
\in \mathbb{R}^{n \times K}$, where the predictors are
$$
z^{\text{BT}}_t = \sum_k c_k (s_{a, k} - s_{b, k}) + \beta, \qquad
z^{\text{M}}_{t, k} = \eta_t c_k (s_{a, k} - s_{b, k}).
$$
The NLL is bilinear in $(c, s)$ and not jointly convex. It can be
solved by reparametrizing in $M = c \odot s$ — the predictors become
affine in $M$ — fitting in $M$, then recovering $(\hat c, \hat s)$ by
$\hat c_k = \|\hat M_{·,k}\|_2 / \sqrt{n}$, $\hat s_{·,k} = \hat M_{·,k} / \hat c_k$.

The parametrization $(s, \gamma, \beta)$ in this package observes the
same dataset structure but has three advantages:

1. **Convex without reparametrization.** The predictor is already affine in
   $(s, \gamma, \beta)$, so no reparametrization is needed. The Hessian is
   PSD by construction, and a single L-BFGS-B run finds the unique optimum
   (we verify via the `inf_diff` perturbed-init sanity check).
2. **Mention intercept $\gamma_k$.** The multiplicative model conflates "how much does
   the judge vary items along criterion $k$" (modulated by $c_k$) with "how
   readily does the judge mention $k$ at all" (no separate parameter). The
   additive model separates these: $\mathrm{std}(\hat s_{·,k})$ and $\hat\gamma_k$
   are independent post-fit summaries with distinct interpretations.
3. **Cleaner identifiability story.** The $M$ parametrization has the
   same $K$-dim per-column null space, broken by the same ridge — but it
   *also* has a sign ambiguity $(c_k, s_{·,k}) \to (-c_k, -s_{·,k})$ that
   needs a post-hoc resolution step. The $(s, \gamma)$ form has no
   analogous sign ambiguity: the sign of $\hat s_{·,k}$ is determined by
   the joint sign of the data (per the §3.3 convention).

Observationally, the two parametrizations produce the same predictions and
hence the same held-out accuracy. The additive form is purely a presentation /
interpretability win, and a convenience for downstream code (no
post-fit sign reconciliation).
