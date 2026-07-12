# Harmonic oscillator: risk-neutral optimal control and statistical inference under uncertainty

A two-state harmonic oscillator whose **angular frequency `k` is uncertain** is
steered to the origin by a two-component control. The frequency is unknown at the
time the control is chosen, so the control is optimized *across the whole
distribution of `k`* with the **sample average approximation (SAA)** — minimizing
the **expected** terminal cost (the risk-neutral problem). The SAA optimal value
computed from a finite scenario sample is itself random, so the demo's main focus
is **quantifying its sampling error**: two confidence-interval algorithms, a
central-limit-theorem study, and a Monte-Carlo test that those intervals attain
their nominal coverage.

The problem is Problem `S_C` of [doi:10.1137/140983161](https://doi.org/10.1137/140983161);
the uncertainty-aware treatment and the statistical results follow Melnikov & Milz
([arXiv:2407.18182](https://arxiv.org/abs/2407.18182)) and the manuscript
*Statistical Inference for Optimal Values in Scenario-Based Optimal Control under
Uncertainty*.

## The model — [harmonic_oscillator.py](harmonic_oscillator.py)

- **State** `x ∈ ℝ²`, **control** `u ∈ ℝ²`, **horizon** `t_f = 1`.
- **Dynamics**: `ẋ₁ = −k·x₂ + u₁`, `ẋ₂ = k·x₁ + u₂`, with initial state
  `x(0) = (1, 0)`. The uncertain angular frequency is `k`.
- **Terminal cost** (minimized): `F(x(t_f)) = ½‖x(t_f)‖²` — how far the final
  state is from the origin.
- **Running cost**: `(α/2)‖u‖²` with `α = 1e-3` (a small control penalty).
- **Control box**: `u ∈ [−3, 3]²`.
- **Uncertainty**: `k ~ U[0, 2π]`, drawn i.i.d. by Monte Carlo
  (`ensemblecontrol.UniformSampler(..., method="mc")`). The **nominal** problem
  fixes `k` at its mean `𝔼[k] = π`.
- **Discretization**: `50` control intervals, **single shooting** (the states are
  integrated out; the decision vector is the piecewise-constant control), RK4.

## Requirements & how to run

Create the project virtualenv once, from the repository root, and install the
`ensemblecontrol` package into it (editable):

```bash
cd <repo-root>/EnsembleControl
python3 -m venv .venv
./.venv/bin/pip install -e .        # installs ensemblecontrol and its dependencies (numpy, casadi, matplotlib, ...)
```

Then run the drivers **from this demo folder** (`demo/harmonic_oscillator/`) so the
relative `../../.venv/bin/python` path resolves and the local `from
harmonic_oscillator import ...` import works. `MPLBACKEND=Agg` renders figures
headless:

```bash
MPLBACKEND=Agg ../../.venv/bin/python saa_harmonic_oscillator.py   # nominal + risk-neutral + confidence intervals
MPLBACKEND=Agg ../../.venv/bin/python clt_harmonic_oscillator.py   # central-limit study + optimization-bias diagnostic
../../.venv/bin/python coverage_harmonic_oscillator.py             # coverage test of the CIs (wrapper: ./run_coverage.sh)
```

Useful flags: `saa_harmonic_oscillator.py --algorithm {plugin,subsampling,both}`
(default `both`), `--m` / `--b` (subsampling count / block size), `--workers`;
`clt_harmonic_oscillator.py --R` (replicates) / `--n-ref` (reference size);
`coverage_harmonic_oscillator.py --ci {plugin,subsampling,both}`, `--R` / `--R-sub`,
`--n-ref`, `--workers` (see [Coverage validation](#coverage-validation) below).
**Every solve uses IPOPT** (`SAAProblem.solve`). Figures use LaTeX when a `latex`
binary is on `PATH`, otherwise matplotlib's mathtext. All outputs land under
[output/](output/) at fixed, timestamp-free paths. The reference solve that proxies
the population optimum (shared by the CLT and coverage studies) is configured once
in [run_config.py](run_config.py).

## Problem formulation

Let `x^u(t_f, ξ)` be the terminal state under control `u` and scenario `ξ = k`,
and `F(x^u(t_f, ξ)) = ½‖x^u(t_f, ξ)‖²` the terminal loss. Both problems add the
running cost `(α/2)∫‖u‖²` and keep `u ∈ [−3, 3]²`.

**Nominal** — solve at the mean parameter only, `k = 𝔼[k] = π` (a single scenario).

**Risk-neutral** — minimize the *expected* terminal loss; with `N` i.i.d.
scenarios `ξ₁,…,ξ_N` the SAA replaces the expectation by the sample mean:

```
min_u  𝔼[F(x^u(t_f, ξ))]      ≈      min_u  (1/N) Σ_i F(x^u(t_f, ξ_i))
```

Representative optimal values from a default run: nominal `J = 4.997e-04`,
risk-neutral `J = 8.518e-03` (the nominal control, tuned to `k = π` alone, is far
more expensive once averaged over the whole `k` distribution).

## Statistical inference

The risk-neutral SAA optimal value `Ĵ_N*` from a finite sample is random. The
package provides two confidence-interval algorithms, a limit-theorem study, and a
coverage test for it (the risk-neutral SAA is solved on nested prefixes
`N ∈ {32, 64, 128}` and reused as the anchor for every algorithm):

- **[saa_harmonic_oscillator.py](saa_harmonic_oscillator.py)** — the **plug-in CI**
  (Algorithm 1: normal interval from the in-sample loss variance, plus
  out-of-sample-variance variants) and the **subsampling CI** (Algorithm 2: valid
  for nonunique optimizers, each subsample re-solved with IPOPT). Raw data go to
  JSON and figures are rendered from it, all timestamp-free, under
  [output/inference/](output/inference/) (`plugin_*`, `plugin-oos_*`,
  `plugin-oos-matched_*`, `subsampling_*`).
- **[clt_harmonic_oscillator.py](clt_harmonic_oscillator.py)** — a Monte-Carlo
  central-limit study: `√N(Ĵ_N* − Ĵ_ref*)` histograms across `N ∈ {32, 64, 128}`,
  each replicate IPOPT-solved and warm-started from an independent size-`N_ref`
  reference solution; output under [output/limit_theorem/](output/limit_theorem/)
  (`clt_*`). It also emits an **optimization-bias diagnostic**
  (`optimization_bias.png`): the per-size mean `𝔼[Ĵ_N*] ± SE` climbing toward the
  reference `Ĵ_ref*` as `N` grows (the optimistic SAA bias, Prop. 5.6).
- **[coverage_harmonic_oscillator.py](coverage_harmonic_oscillator.py)** — a
  Monte-Carlo **coverage test** that both CIs actually attain their nominal
  coverage, reporting a rigorous lower bound on the true coverage probability. See
  [Coverage validation](#coverage-validation) below.

## Coverage validation

A confidence interval is only useful if it covers the truth as often as it claims.
[coverage_harmonic_oscillator.py](coverage_harmonic_oscillator.py) runs a
Monte-Carlo **coverage test** of both CIs and reports a *rigorous lower bound* on
their true coverage probability — the empirical coverage-validation loop of Eichhorn
& Römisch (2007), §6, whose CI algorithms this package implements.

Because `J* = v(P)` has no closed form, it is proxied by `Ĵ_ref* = f_ref`, the SAA
optimal value on one large independent **reference sample** of size `N_ref` (default
`4096`); its `O(1/√N_ref)` error is negligible next to the CI half-width `O(1/√N)`.
This is the same reference proxy the CLT study uses — both read `N_ref`, the root
seed, and the reference tolerance from [run_config.py](run_config.py), so their
`Ĵ_ref*` cannot drift apart.

For each `N ∈ {32, 64, 128}` and `R` replications (common random numbers across `N`;
the reference stream stays independent): solve the SAA, build the CI, and record the
indicator `Z_j = 1{lo_j ≤ f_ref ≤ hi_j}`. With `L = Σ_j Z_j`, the empirical coverage
is `L/R`, and the estimator in
[probability_estimator.py](../../src/ensemblecontrol/probability_estimator.py),

```
p̂_{R,δ}(L) = min{ q ∈ [0,1] : Σ_{k=L}^{R} C(R,k) q^k (1−q)^{R−k} ≥ δ }
           = scipy.stats.beta.ppf(δ, L, R−L+1),
```

is the **Clopper–Pearson lower confidence bound**: with confidence `1−δ` the true
coverage `p` is at least `p̂_{R,δ}(L)` (Nemirovski §10.2.1, eq. 10.2.4 / Lemma
10.2.1). `δ` (default `1e-6`) is the failure probability of *this bound* — not the
CI's own level `1−β`. The gap `L/R − p̂` shrinks like `1/√R`, so the plug-in run
uses a large `R` to make the guarantee tight.

**Reading the table.** Each row is a size `N`; each nominal level `1−β` shows `L/R`
(empirical coverage) and `p̲_δ = p̂_{R,δ}(L)` (the lower bound). If `p̲_δ ≥ 1−β`,
then with confidence `1−δ` the CI is *not* anti-conservative at that level; if
`p̲_δ` sits well below `1−β`, the CI under-covers.

| CI | interval built from | solves per `N` |
|----|---------------------|----------------|
| **plug-in** (Alg. 1) | one extra control rollout at the optimizer, no re-solve | `R + 1` |
| **subsampling** (Alg. 2) | re-solve `m` subsamples of size `b = default_subsample_size(N)`, warm-started | `R·(m+1) + 1` |

Subsampling is far more expensive, so it defaults to a smaller `R` (`--R-sub 200`,
`--m-sub 200`); choose the CI with `--ci {plugin,subsampling,both}`. The `R`
replicate solves per `N` run **in parallel while each solve is serial** (default
`cpu−2` workers); samples are pre-drawn from independent streams and results
collected in input order, so the indicators are **bit-identical for any worker
count**. (`--workers auto` collapses to a single outer thread here — each `N`
already exceeds `cpu−2` — so pass an integer for true replicate-level parallelism.)

Outputs go to [output/coverage/](output/coverage/) as
`coverage_<ci>.{json,tex,txt}` (`<ci>` ∈ {`plugin`, `subsampling`}). The JSON stores
only the raw indicators; the empirical coverage and lower bounds are recomputed at
report time (`coverage_from_indicators`, `coverage_latex_table`), so a table can be
re-derived — e.g. for a different `δ` — without re-running the study:

```python
import ensemblecontrol
print(ensemblecontrol.coverage_latex_table("output/coverage/coverage_plugin.json",
                                           deltas=(0.05, 0.10)))
```

References: A. Nemirovski, §10.2.1 ("Simulation-Based Justification"),
<https://www2.isye.gatech.edu/~nemirovs/FullBookDec11.pdf>; A. Eichhorn & W.
Römisch, *Stochastic Integer Programming: Limit Theorems and Confidence Intervals*,
Math. Oper. Res. 32(1):118–135, 2007 (§5 the CI algorithms, §6 the coverage loop).

## Plots

Control and state trajectories for the nominal and risk-neutral solves are written
to [output/controls-state/](output/controls-state/) as
`{nominal,risk-neutral}_{controls,states}.png`. The control plots are framed on the
`[−3, 3]` box; the state plots show the ensemble mean `𝔼[xⱼ]` with a ±1 s.d. band.

Optimal controls `u₁*(t)`, `u₂*(t)` and the ensemble states `𝔼[xⱼ*(t, ξ)]`:

| risk-neutral controls | risk-neutral states |
| --- | --- |
| ![](output/controls-state/risk-neutral_controls.png) | ![](output/controls-state/risk-neutral_states.png) |

## Scripts

| Script | Solver | What it does |
| --- | --- | --- |
| [harmonic_oscillator.py](harmonic_oscillator.py) | — | The model (`HarmonicOscillator(ControlProblem)`); imported, not run directly. |
| [saa_harmonic_oscillator.py](saa_harmonic_oscillator.py) | IPOPT | Nominal + risk-neutral solves and control/state plots, then the plug-in (Algorithm 1) and subsampling (Algorithm 2) confidence intervals for the risk-neutral SAA optimal value. |
| [clt_harmonic_oscillator.py](clt_harmonic_oscillator.py) | IPOPT | Monte-Carlo central-limit-theorem study: `√N(Ĵ_N* − J*)` histograms across `N ∈ {32, 64, 128}`, each replicate IPOPT-solved and warm-started from an independent size-`N_ref` reference solution; also the optimization-bias diagnostic. |
| [coverage_harmonic_oscillator.py](coverage_harmonic_oscillator.py) | IPOPT | Monte-Carlo coverage test of both CIs against the size-`N_ref` reference value, reporting the Clopper–Pearson lower bound on the true coverage; `--ci {plugin,subsampling,both}`. |
| [run_config.py](run_config.py) | — | Shared reference-solve configuration (`N_ref`, root seed, tolerance) for the CLT and coverage studies; imported, not run directly. |

## Notes

- **IPOPT throughout** — the nominal/risk-neutral solves, the subsampling re-solves,
  and the CLT replicates all use IPOPT (`SAAProblem.solve`). Warm-started solves
  project the starting control into the strictly-interior box
  `[lb + 0.01·(ub−lb), ub − 0.01·(ub−lb)] = [−2.94, 2.94]`, since an interior-point
  method must not start exactly on a bound.
- **Confidence intervals require i.i.d. scenarios**, so `k` is drawn by Monte Carlo
  (`method="mc"`); the full-sample (`N = 128`) SAA is solved once and reused as the
  largest plug-in sample size and the subsampling anchor.
- **Fixed, timestamp-free outputs.** Re-running overwrites the same files under
  [output/](output/).
- CVaR risk-averse control is available in the framework (`SAAProblem(..., beta=β)`)
  and demonstrated in the [batch-reactor demo](../batch_reactor/); for this
  oscillator the risk-averse control differs little from the risk-neutral one, so it
  is not included here.
