"""Monte-Carlo coverage test of the SAA confidence intervals for the harmonic
oscillator -- for BOTH the plug-in (Algorithm 1) and subsampling (Algorithm 2)
confidence intervals.

For the harmonic oscillator of Melnikov & Milz (arXiv:2407.18182), with uncertain
angular frequency k ~ U[0, 2*pi], a confidence interval for the SAA optimal value
is meant to cover the population optimal value J* with its nominal probability
1 - alpha.  J* is not computable, so we proxy it by J_hat_ref*, the SAA value on
one large independent reference sample of size N_ref.

For each N in {32, 64, 128} we run R independent replications (fresh i.i.d.
scenario draws), build the CI on each, and count the L that cover J_hat_ref*.  The
empirical coverage is L/R; the estimator ``probability_lower_bound(R, L, delta)``
upgrades it to a rigorous (1-delta) lower confidence bound on the true coverage
(Clopper-Pearson; eq. 10.2.4 / Lemma 10.2.1).  This is the empirical
coverage-validation loop of Eichhorn & Roemisch (2007), Sec. 6, reporting the
guaranteed lower bound rather than the raw ratio.  See README_coverage.md.

Two confidence intervals are validated (choose with --ci):
  * plug-in (Algorithm 1): one extra rollout per replication, no re-solve, so the
    cost is exactly R + 1 SAA solves per N.
  * subsampling (Algorithm 2): each replication re-solves m subsamples of size
    b = default_subsample_size(N), so the cost is R*(m + 1) + 1 solves per N (much
    larger -- use a smaller --R-sub).  The m re-solves run serially inside each
    replication (``workers=1``) so the outer replicate threads stay saturated.

The R replicate solves per N run in parallel (default: cpu-2 workers) while each
solve stays serial.  The raw coverage indicators are saved per CI to
output/coverage/coverage_<ci>.json and the LaTeX table is rendered from them via
``ensemblecontrol.coverage_latex_table``, so a table can be re-derived without
re-running the study.  This is model-agnostic: any model + ensemblecontrol sampler
reuses ``coverage_study`` by supplying a solve callback (and, for subsampling, a
single-shooting box-only problem so the default scipy re-solver applies).

Usage (from this directory; prefix with MPLBACKEND=Agg on headless machines):
    ../../.venv/bin/python coverage_harmonic_oscillator.py
    ... --ci plugin                       # only the plug-in CI
    ... --ci subsampling --R-sub 100      # only subsampling, cheaper
    ... --ci plugin --R 100 --n-ref 512   # cheap smoke run
"""

import argparse
import os
import sys

import numpy as np

import ensemblecontrol
from ensemblecontrol.inference import _BUILD_LOCK, _core_budget  # CasADi lock; cpu-2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harmonic_oscillator import HarmonicOscillator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
COVERAGE_DIR = os.path.join(HERE, "output", "coverage")
LEVELS = (0.90, 0.95, 0.99)     # nominal CI confidence levels 1 - alpha

# Human-readable labels for the LaTeX caption, keyed by the --ci name.
CI_LABEL = {"plugin": "plug-in confidence interval (Algorithm 1)",
            "subsampling": "subsampling confidence interval (Algorithm 2)"}


def strict_interior(w, lb=-3.0, ub=3.0, margin=0.01):
    # Clip a warm start into the control box shrunk by a range-relative margin so
    # it is strictly feasible for IPOPT's interior-point method.
    span = ub - lb
    return np.clip(w, lb + margin * span, ub - margin * span)


def make_ipopt_solve(model, tol=1e-5):
    # A ready-made solve(samples, w0=None, inner_serial=False) -> (saa, w_opt, f_opt)
    # callback, using IPOPT (single shooting, box [-3,3]). Construction is serialized
    # under _BUILD_LOCK and the solve runs outside it, so the callback is safe under
    # the study's replicate threads; inner_serial pins the per-sample map serial when
    # those replicate solves are themselves threaded. A warm start w0 is projected
    # strictly interior for the interior-point method.
    quiet = {"print_level": 0, "sb": "yes"}   # silence IPOPT (thousands of solves)

    def solve(samples, w0=None, inner_serial=False):
        kw = dict(parallelization="serial", n_threads=1) if inner_serial else {}
        with _BUILD_LOCK:
            saa = ensemblecontrol.SAAProblem(model, samples, MultipleShooting=False,
                                             tol=tol, ipopt_options=quiet, **kw)
            if w0 is not None:
                saa.initial_decisions = strict_interior(np.asarray(w0, dtype=float))
        w_opt, f_opt = saa.solve()
        return saa, w_opt, float(np.ravel(f_opt)[0])   # solve() returns f as a 1-elem array
    return solve


def make_subsampling_ci_of(m, levels, scipy_tol=1e-5, seed=0):
    # A ci_of(saa, w_opt, f_opt) -> ci that builds the subsampling CI (Algorithm 2):
    # re-solve m subsamples of size b = default_subsample_size(N) with the default
    # scipy L-BFGS-B re-solver (single-shooting, box-only), warm-started from w_opt.
    # A fresh default_rng(seed) is created per call so the m subsample index sets are
    # deterministic and thread-safe (the scenarios differ per replication, so the
    # subsample CONTENT still varies); workers=1 keeps the m re-solves serial inside
    # the outer replicate thread.
    def ci_of(saa, w_opt, f_opt):
        b = ensemblecontrol.default_subsample_size(saa.nsamples)
        rng = np.random.default_rng(seed)
        rec = ensemblecontrol.subsampling_confidence_interval(
            saa, f_opt, b=b, m=m, rng=rng, w_opt=w_opt, levels=levels,
            scipy_tol=scipy_tol, workers=1)
        return rec["ci"]
    return ci_of


# -- study parameters --------------------------------------------------------
NS = (32, 64, 128)     # training sample sizes whose CI is validated
N_REF = 4096           # independent reference sample size (proxies J*), kept large
R_PLUGIN = 5000        # plug-in replications (1 solve each -> cheap)
R_SUB = 200            # subsampling replications (m+1 solves each -> expensive)
M_SUB = 200            # subsamples re-solved per subsampling replication
ROOT_SEED = 12345      # root entropy; independent child streams are spawned from it


def run_one(name, ci_of, R, solve, args, run_dir):
    # A fresh root sampler (same seed) per CI -> the reference value f_ref and the
    # replicate draws are identical across the two analyses, so plug-in and
    # subsampling coverage are a paired comparison on the same data.
    root = ensemblecontrol.UniformSampler(0.0, 2.0 * np.pi, method="mc",
                                          seed=ROOT_SEED)

    def progress(N, done, total):
        if done % max(1, total // 10) == 0 or done == total:
            print("  [%s] N=%3d  replication %5d/%d" % (name, N, done, total))

    study = ensemblecontrol.coverage_study(
        root, solve, sample_sizes=NS, R=R, n_ref=args.n_ref, levels=LEVELS,
        ci_of=ci_of, workers=args.workers, progress=progress)

    print("[%s] reference J_hat_ref* (N_ref=%d) = %.8f"
          % (name, study["n_ref"], study["f_ref"]))

    path = ensemblecontrol.save_coverage_run(
        study, os.path.join(run_dir, "coverage_%s.json" % name),
        meta={"model": "HarmonicOscillator", "ci": name,
              "sampler": "UniformSampler", "k_distribution": "U[0, 2*pi]",
              "seed": ROOT_SEED})

    caption = ("Estimated coverage of the %s for the SAA optimal value of the "
               "harmonic oscillator. For each training size $N$ and nominal level "
               "$1-\\alpha$, $L/R$ is the empirical coverage over $R=%d$ "
               "replications and $\\underline{p}_{\\delta}$ is the $(1-\\delta)$ "
               "lower confidence bound $\\hat p_{R,\\delta}(L)$."
               % (CI_LABEL[name], study["R"]))
    tex = ensemblecontrol.coverage_latex_table(
        path, deltas=args.deltas, caption=caption, label="tab:coverage_%s" % name)
    with open(os.path.join(run_dir, "coverage_%s.tex" % name), "w") as fh:
        fh.write(tex + "\n")

    print(tex)
    return study


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ci", default="both",
                    choices=["plugin", "subsampling", "both"],
                    help="which confidence interval(s) to validate (default both)")
    ap.add_argument("--R", type=int, default=R_PLUGIN,
                    help="plug-in replications per N (1 solve each)")
    ap.add_argument("--R-sub", type=int, default=R_SUB,
                    help="subsampling replications per N (m+1 solves each)")
    ap.add_argument("--m-sub", type=int, default=M_SUB,
                    help="subsamples re-solved per subsampling replication")
    ap.add_argument("--n-ref", type=int, default=N_REF,
                    help="reference sample size (proxies J*)")
    ap.add_argument("--deltas", default="0.05",
                    help="comma-separated failure probabilities for the lower "
                         "bound (delta=0.05 -> 95%%-confident lower bound)")
    ap.add_argument("--workers", default=str(_core_budget()),
                    help="outer parallelism over the R replicate solves per N: "
                         "default cpu-2 (= %d here) runs that many solves at once, "
                         "each solve serial (single-threaded) -- the intended "
                         "'solves in parallel, each serial' mode. Pass an integer "
                         "to override, 1 for a sequential (inner-threaded) loop, or "
                         "'auto' to let the study pick (collapses to 1 when a "
                         "single size-N solve already saturates the cores)."
                         % _core_budget())
    args = ap.parse_args()
    args.deltas = tuple(float(d) for d in args.deltas.split(","))

    run_dir = COVERAGE_DIR   # fixed, timestamp-free: output/coverage/
    model = HarmonicOscillator()
    # IPOPT on the control box [-3, 3] (single shooting); each replicate is
    # warm-started (strictly interior) from the reference solution inside the study.
    # tol=1e-5 is the *inference* tolerance: with thousands of small solves a tight
    # 1e-8 wastes ~3x the IPOPT iterations, and J_hat_N* shifts by ~1e-5 << CI width.
    solve = make_ipopt_solve(model, tol=1e-5)

    if args.ci in ("plugin", "both"):
        run_one("plugin", None, args.R, solve, args, run_dir)   # default = plug-in CI
    if args.ci in ("subsampling", "both"):
        ci_of = make_subsampling_ci_of(args.m_sub, LEVELS)
        run_one("subsampling", ci_of, args.R_sub, solve, args, run_dir)


if __name__ == "__main__":
    main()
