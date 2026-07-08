"""Monte-Carlo coverage test of the plug-in confidence interval for the SAA value.

For the harmonic oscillator of Melnikov & Milz (arXiv:2407.18182), with uncertain
angular frequency k ~ U[0, 2*pi], the plug-in interval (Algorithm 1) is meant to
cover the population optimal value J* with its nominal probability 1 - alpha.  J*
is not computable, so we proxy it by J_hat_ref*, the SAA value on one large
independent reference sample of size N_ref.

For each N in {32, 64, 128} we run R independent replications (fresh i.i.d.
scenario draws), build the plug-in CI on each, and count the L that cover
J_hat_ref*.  The empirical coverage is L/R; the estimator
``probability_lower_bound(R, L, delta)`` upgrades it to a rigorous (1-delta) lower
confidence bound on the true coverage (Clopper-Pearson; eq. 10.2.4 / Lemma 10.2.1).
This is the empirical coverage-validation loop of Eichhorn & Roemisch (2007), Sec. 6,
reporting the guaranteed lower bound rather than the raw ratio.

Each replication is one SAA solve; the R solves per N run in parallel while each
solve stays serial (``inner_serial`` when the outer loop is threaded).  It uses
IPOPT on the control box [-3, 3] (single shooting) and warm-starts every replicate
from the reference solution (projected strictly interior), so the many small solves
converge quickly.  The raw coverage indicators are saved to
output/coverage/coverage.json and the LaTeX table is rendered from them via
``ensemblecontrol.coverage_latex_table``, so the table can be re-derived without
re-running the study.  This is model-agnostic: any model + ensemblecontrol sampler
reuses ``coverage_study`` by supplying a solve callback.

Usage (from this directory; prefix with MPLBACKEND=Agg on headless machines):
    ../../.venv/bin/python coverage_harmonic_oscillator.py
    ... --R 100 --n-ref 512          # cheap smoke run
"""

import argparse
import os
import sys

import numpy as np

import ensemblecontrol
from ensemblecontrol.inference import _BUILD_LOCK   # serialize CasADi construction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harmonic_oscillator import HarmonicOscillator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
COVERAGE_DIR = os.path.join(HERE, "output", "coverage")


def strict_interior(w, lb=-3.0, ub=3.0, margin=0.01):
    # Clip a warm start into the control box shrunk by a range-relative margin so
    # it is strictly feasible for IPOPT's interior-point method.
    span = ub - lb
    return np.clip(w, lb + margin * span, ub - margin * span)


def make_ipopt_solve(model, tol=1e-8):
    # A ready-made solve(samples, w0=None, inner_serial=False) -> (saa, w_opt, f_opt)
    # callback for coverage_study, using IPOPT (single shooting, box [-3,3]).
    # Construction is serialized under _BUILD_LOCK and the solve runs outside it, so
    # the callback is safe under the study's replicate threads; inner_serial pins the
    # per-sample map serial when those replicate solves are themselves threaded. A
    # warm start w0 is projected strictly interior for the interior-point method.
    # Silence IPOPT's per-solve log/banner -- the study runs thousands of solves.
    quiet = {"print_level": 0, "sb": "yes"}

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


# -- study parameters --------------------------------------------------------
NS = (32, 64, 128)     # training sample sizes whose plug-in CI is validated
N_REF = 4096           # independent reference sample size (proxies J*), kept large
R = 5000               # replications per N (large so L/R -> p_lower gap ~ 1/sqrt(R))
ROOT_SEED = 12345      # root entropy; independent child streams are spawned from it


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--R", type=int, default=R, help="replications per N")
    ap.add_argument("--n-ref", type=int, default=N_REF, help="reference sample size")
    ap.add_argument("--deltas", default="0.05",
                    help="comma-separated failure probabilities for the lower "
                         "bound (delta=0.05 -> 95%%-confident lower bound)")
    ap.add_argument("--workers", default="auto",
                    help="parallelism over the R replicate solves per N: 'auto' "
                         "(default; no outer threads when a size-N solve already "
                         "saturates the cores), 1, or an integer worker count")
    args = ap.parse_args()
    deltas = tuple(float(d) for d in args.deltas.split(","))

    run_dir = COVERAGE_DIR   # fixed, timestamp-free: output/coverage/

    model = HarmonicOscillator()
    # Reproducible i.i.d. scenario root (k ~ U[0, 2*pi]); coverage_study splits it
    # into the reference stream plus one group per sample size (each spawning R
    # replicate streams).
    root = ensemblecontrol.UniformSampler(0.0, 2.0 * np.pi, method="mc",
                                          seed=ROOT_SEED)
    # IPOPT on the control box [-3, 3] (single shooting); each replicate is
    # warm-started (strictly interior) from the reference solution inside the study.
    solve = make_ipopt_solve(model, tol=1e-8)

    def progress(N, done, total):
        if done % 250 == 0 or done == total:
            print("  N=%3d  replication %4d/%d" % (N, done, total))

    study = ensemblecontrol.coverage_study(
        root, solve, sample_sizes=NS, R=args.R, n_ref=args.n_ref,
        workers=args.workers, progress=progress)

    print("reference J_hat_ref* (N_ref=%d) = %.8f" % (study["n_ref"], study["f_ref"]))

    # q = mesh size = number of control intervals (= len of the control vector)
    path = ensemblecontrol.save_coverage_run(
        study, os.path.join(run_dir, "coverage.json"),
        meta={"model": "HarmonicOscillator", "sampler": "UniformSampler",
              "k_distribution": "U[0, 2*pi]", "seed": ROOT_SEED})

    caption = ("Estimated coverage of the plug-in confidence interval for the SAA "
               "optimal value of the harmonic oscillator. For each training size "
               "$N$ and nominal level $1-\\alpha$, $L/R$ is the empirical coverage "
               "over $R={}$ replications and $\\underline{{p}}_{{\\delta}}$ is the "
               "$(1-\\delta)$ lower confidence bound "
               "$\\hat p_{{R,\\delta}}(L)$.").format(study["R"])
    tex = ensemblecontrol.coverage_latex_table(
        path, deltas=deltas, caption=caption, label="tab:coverage")
    with open(os.path.join(run_dir, "coverage.tex"), "w") as fh:
        fh.write(tex + "\n")

    print(tex)


if __name__ == "__main__":
    main()
