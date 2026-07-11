# Coverage analysis of the SAA confidence intervals (harmonic oscillator)

`coverage_harmonic_oscillator.py` runs a Monte-Carlo **coverage test**: it checks
whether the confidence intervals (CIs) that `ensemblecontrol` builds for the SAA
optimal value actually attain their nominal coverage, and reports a *rigorous
lower bound* on the true coverage probability. Both CIs are validated:

- the **plug-in** CI (Algorithm 1, `plugin_confidence_interval`), and
- the **subsampling** CI (Algorithm 2, `subsampling_confidence_interval`).

This is the empirical coverage-validation loop of Eichhorn & Römisch (2007), §6 —
whose CI algorithms this package implements — reporting the guaranteed lower bound
instead of the raw ratio.

## What "coverage" means here

A CI rule for the population optimal value `J*` is *valid at level 1−α* if the
random interval `[lo, hi]` (random through the sample) contains `J*` with
probability at least `1−α`. That true coverage probability `p` is unknown; the
coverage test estimates it, and — crucially — bounds it from below with
confidence.

Because `J* = v(P)` has no closed form for this problem, we proxy it by
**`J_hat_ref* = f_ref`**, the SAA optimal value on one large, independent
**reference sample** of size `N_ref` (default 4096). Its `O(1/√N_ref)` error is
negligible next to the CI half-width `O(1/√N)` when `N_ref ≫ N`. The same `f_ref`
is used for every `N` (the population optimum does not depend on the training
size). This is exactly the reference-value proxy the CLT study uses.

## The procedure

For each training size `N ∈ {32, 64, 128}`:

1. Draw `R` independent i.i.d. scenario samples of size `N` (fresh streams spawned
   from a single seeded root, so the run is reproducible).
2. For replication `j`: solve the SAA, build the CI, and record the **indicator**
   `Z_j = 1{ lo_j ≤ f_ref ≤ hi_j }`.
3. Let `L = Σ_j Z_j` be the number of covering intervals over `R` replications.

The indicators `Z_1, …, Z_R` are i.i.d. `Bernoulli(p)`, where `p` is the CI rule's
true coverage — the empirical coverage is the ratio `L/R`.

## From the count to a guaranteed lower bound (the estimator)

We turn `L/R` into a rigorous statement with the estimator in
[`probability_estimator.py`](../../src/ensemblecontrol/probability_estimator.py),

    p̂_{R,δ}(L) = min{ q ∈ [0,1] : Σ_{k=L}^{R} C(R,k) q^k (1−q)^{R−k} ≥ δ },

which is the **Clopper–Pearson lower confidence bound** on a binomial success
probability (eq. 10.2.4 / Lemma 10.2.1). The tail sum equals the regularized
incomplete beta `I_q(L, R−L+1)`, so `p̂ = scipy.stats.beta.ppf(δ, L, R−L+1)`
(with `p̂(0)=0`, `p̂(R)=δ^{1/R}`). By Lemma 10.2.1,

    Prob{ p̂_{R,δ}(L) > p } ≤ δ,

i.e. **with confidence `1−δ`, the true coverage `p` is at least `p̂_{R,δ}(L)`**.
`δ` is the failure probability of *this bound* (default `δ=0.05` ⇒ a 95%-confident
lower bound) — not the CI's own confidence level `1−α`.

The gap between the point estimate `L/R` and the guaranteed bound `p̂` shrinks like
`1/√R`, which is why the plug-in run uses `R=5000`: a large `R` makes the
guarantee tight.

**Reading the output table.** Each row is a training size `N`; each nominal level
`1−α` shows `L/R` (empirical coverage) and `p̲_δ = p̂_{R,δ}(L)` (the lower bound).
If `p̲_δ ≥ 1−α`, then with confidence `1−δ` the CI is *not* anti-conservative at
that level; if `p̲_δ` sits well below `1−α`, the CI under-covers.

## Plug-in vs. subsampling — and cost

| CI | how the interval is built | solves per `N` |
|----|---------------------------|----------------|
| **plug-in** (Alg. 1) | one extra control rollout at the optimizer (`terminal_losses`), no re-solve | `R + 1` |
| **subsampling** (Alg. 2) | re-solve `m` subsamples of size `b = default_subsample_size(N)`, warm-started | `R·(m+1) + 1` |

Subsampling is far more expensive, so it defaults to a smaller `R` (`--R-sub 200`,
`--m-sub 200`). The `m` subsample re-solves of a replication run **serially**
(`workers=1`) so the outer replicate threads stay saturated. The subsample index
sets are drawn from a fresh `default_rng(seed)` per replication — deterministic and
thread-safe; the scenarios differ per replication, so the subsample *content* still
varies.

## Parallelism

The `R` replicate solves per `N` run **in parallel while each solve is serial**.
`coverage_study` threads the replicate loop with up to **`cpu_count − 2`** workers
(`_core_budget()`, the demo's default `--workers`) and forces each solve
single-threaded (`inner_serial=True`, i.e. the CasADi map is built
`parallelization="serial", n_threads=1`). CasADi graph *construction* is serialized
under a build lock; the IPOPT/scipy solves run outside it and release the GIL, so
they genuinely overlap. Because all replicate samples are pre-drawn from independent
streams and results are collected in input order, the indicators are **bit-identical
for any worker count** (verified in `test/test_coverage.py`).

> Note: `--workers auto` would collapse to a *single* outer thread here, because
> each `N ∈ {32,64,128}` already exceeds `cpu−2`, so "auto" assumes one solve
> saturates the cores and threads that solve's inner map instead. Pass an integer
> (the default) for true replicate-level parallelism.

## Running it

From this directory (use the repo virtualenv; `MPLBACKEND=Agg` on headless
machines), or via the wrapper `./run_coverage.sh`:

```bash
../../.venv/bin/python coverage_harmonic_oscillator.py                 # both CIs
../../.venv/bin/python coverage_harmonic_oscillator.py --ci plugin     # plug-in only
../../.venv/bin/python coverage_harmonic_oscillator.py --ci subsampling --R-sub 100
../../.venv/bin/python coverage_harmonic_oscillator.py --ci plugin --R 100 --n-ref 512  # smoke
```

Flags: `--ci {plugin,subsampling,both}`, `--R` (plug-in reps), `--R-sub`/`--m-sub`
(subsampling reps / subsamples), `--n-ref`, `--deltas` (comma-separated), `--workers`.

## Outputs (file names indicate which CI)

Written to `output/coverage/`, with `<ci>` ∈ {`plugin`, `subsampling`}:

| file | contents |
|------|----------|
| `coverage_<ci>.json` | **raw** per-`N` per-level coverage indicators + `f_ref`, `N_ref`, `levels`, `R` |
| `coverage_<ci>.tex`  | the publication `booktabs` table (`\usepackage{booktabs}`) |
| `coverage_<ci>.txt`  | a human-readable summary (`L/R` and the lower bound at `δ=0.05`) |

The JSON stores only the raw indicators; the empirical coverage and the lower
bounds are recomputed from them at report time (`coverage_from_indicators`,
`coverage_latex_table`), so a table can be re-derived — e.g. for a different `δ` —
without re-running the study:

```python
import ensemblecontrol
print(ensemblecontrol.coverage_latex_table("output/coverage/coverage_plugin.json",
                                           deltas=(0.05, 0.10)))
```

## Library entry points

- `ensemblecontrol.coverage_study(sampler, solve, sample_sizes, R, n_ref, ci_of=None, …)`
  — the driver (`ci_of=None` ⇒ plug-in; pass a callback for any other CI).
- `ensemblecontrol.probability_lower_bound(n, ell, delta)` / `binomial_upper_tail`
  — the estimator.
- `ensemblecontrol.coverage_from_indicators`, `save_coverage_run`,
  `load_coverage_run`, `coverage_latex_table` — aggregation, persistence, table.

## References

- The estimator `p̂_{R,δ}(L)` — eq. (10.2.4) and Lemma 10.2.1 — is from
  A. Nemirovski, section 10.2.1 ("Simulation-Based Justification"),
  <https://www2.isye.gatech.edu/~nemirovs/FullBookDec11.pdf>.
- A. Eichhorn and W. Römisch, *Stochastic Integer Programming: Limit Theorems and
  Confidence Intervals*, Mathematics of Operations Research 32(1):118–135, 2007
  (§5 the CI algorithms, §6 the empirical coverage-validation loop).
