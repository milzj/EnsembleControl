"""Monte-Carlo illustration of the statistical limit theorem for the SAA value.

For the harmonic oscillator of Melnikov & Milz (arXiv:2407.18182), with uncertain
angular frequency k ~ U[0, 2*pi], the theory predicts

    N^{1/2}(J_hat_N* - J*)  =>  centered Gaussian     (unique optimizer),

where J_hat_N* is the SAA optimal value on N scenarios and J* is the population
optimal value.  J* is not computable, so we proxy it by J_hat_ref*, the SAA value
on an independent reference sample of size N_ref.

For each N in {32, 64, 128} we solve R independent SAA problems (fresh i.i.d.
scenario draws), record J_hat_N*, and histogram the statistic
sqrt(N)*(J_hat_N* - J_hat_ref*).  As N grows the histogram should look
increasingly Gaussian.

It uses IPOPT on the control box [-3,3] (single shooting) and warm-starts every
replicate from the reference solution (projected strictly interior), so the many
small solves converge quickly.  The raw replicate values are saved to
output/limit_theorem/clt.json and the histograms are rendered from them via
``ensemblecontrol.plot_clt``, so the figures can be re-drawn without re-running
the study.

Usage (from this directory; prefix with MPLBACKEND=Agg on headless machines):
    ../../.venv/bin/python clt_harmonic_oscillator.py
    ... --R 100 --n-ref 512          # cheaper run
"""

import argparse
import os
import sys

import numpy as np

import ensemblecontrol
from ensemblecontrol.inference import _BUILD_LOCK   # serialize CasADi construction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harmonic_oscillator import HarmonicOscillator

HERE = os.path.dirname(os.path.abspath(__file__))
CLT_DIR = os.path.join(HERE, "output", "limit_theorem")


def strict_interior(w, lb=-3.0, ub=3.0, margin=0.01):
    # Clip a warm start into the control box shrunk by a range-relative margin so
    # it is strictly feasible for IPOPT's interior-point method.
    span = ub - lb
    return np.clip(w, lb + margin * span, ub - margin * span)


def make_ipopt_solve(model, tol=1e-8):
    # A ready-made solve(samples, w0=None, inner_serial=False) -> (saa, w_opt, f_opt)
    # callback for clt_replication_study, using IPOPT (single shooting, box [-3,3]).
    # Construction is serialized under _BUILD_LOCK and the solve runs outside it, so
    # the callback is safe under the study's replicate threads; inner_serial pins the
    # per-sample map serial when those replicate solves are themselves threaded. A
    # warm start w0 is projected strictly interior for the interior-point method.
    def solve(samples, w0=None, inner_serial=False):
        kw = dict(parallelization="serial", n_threads=1) if inner_serial else {}
        with _BUILD_LOCK:
            saa = ensemblecontrol.SAAProblem(model, samples, MultipleShooting=False,
                                             tol=tol, **kw)
            if w0 is not None:
                saa.initial_decisions = strict_interior(np.asarray(w0, dtype=float))
        w_opt, f_opt = saa.solve()
        return saa, w_opt, float(np.ravel(f_opt)[0])   # solve() returns f as a 1-elem array
    return solve

# -- study parameters --------------------------------------------------------
NS = (32, 64, 128)     # sample sizes for the statistic
N_REF = 1024           # independent reference sample size (proxies J*)
R = 200                # replicate SAA solves per sample size
ROOT_SEED = 12345      # root entropy; independent child streams are spawned from it


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--R", type=int, default=R, help="replicate solves per N")
    ap.add_argument("--n-ref", type=int, default=N_REF, help="reference sample size")
    ap.add_argument("--workers", default="auto",
                    help="parallelism over the R replicate solves per N: 'auto' "
                         "(default; no outer threads when a size-N solve already "
                         "saturates the cores), 1, or an integer worker count")
    args = ap.parse_args()

    run_dir = CLT_DIR   # fixed, timestamp-free: output/limit_theorem/

    model = HarmonicOscillator()
    # Reproducible i.i.d. scenario root (k ~ U[0, 2*pi]); clt_replication_study
    # splits it into the reference stream plus one group per sample size (each
    # spawning R replicate streams).
    root = ensemblecontrol.UniformSampler(0.0, 2.0 * np.pi, method="mc",
                                          seed=ROOT_SEED)
    # IPOPT on the control box [-3, 3] (single shooting); each replicate is
    # warm-started (strictly interior) from the reference solution inside the study.
    solve = make_ipopt_solve(model, tol=1e-8)

    def progress(N, done, total):
        if done % 50 == 0 or done == total:
            print("  N=%3d  replicate %3d/%d" % (N, done, total))

    study = ensemblecontrol.clt_replication_study(
        root, solve, sample_sizes=NS, R=args.R, n_ref=args.n_ref,
        workers=args.workers, progress=progress)

    print("reference J_hat_ref* (N_ref=%d) = %.8f" % (study["n_ref"], study["f_ref"]))
    for N in NS:
        stat = ensemblecontrol.clt_statistic(study["values_by_N"][N], N,
                                             study["f_ref"])
        print("  N=%3d  mean=%.4f std=%.4f" % (N, stat.mean(), stat.std()))

    # q = mesh size = number of control intervals (= len of the control vector)
    path = ensemblecontrol.save_clt_run(
        study["values_by_N"], os.path.join(run_dir, "clt.json"),
        N_ref=study["n_ref"], f_ref=study["f_ref"], q=study["q"],
        meta={"model": "HarmonicOscillator", "sampler": "UniformSampler",
              "k_distribution": "U[0, 2*pi]", "seed": ROOT_SEED})
    ensemblecontrol.plot_clt(path, outdir=run_dir, stamp="")   # from saved JSON
    # optimization-bias diagnostic: mean SAA optimal value E[Jhat_N*] + reference
    ensemblecontrol.plot_optimization_bias(path, outdir=run_dir)


if __name__ == "__main__":
    main()
