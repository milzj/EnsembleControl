"""Monte-Carlo illustration of the statistical limit theorem for the SAA value.

For the batch reactor of Terwiesch & Agarwal (1995) with uncertain decomposition
collision factor k20 ~ truncnorm(1000, 500) on [500, 2000], the theory predicts

    N^{1/2}(J_hat_N* - J*)  =>  centered Gaussian     (unique optimizer),

where J_hat_N* = E_N[-B] is the risk-neutral SAA optimal value on N scenarios and
J* is the population optimal value.  J* is not computable, so we proxy it by
J_hat_ref*, the SAA value on an independent reference sample of size N_ref.

For each N in {32, 64, 128} we solve R independent SAA problems (fresh i.i.d.
scenario draws), record J_hat_N*, and histogram sqrt(N)*(J_hat_N* - J_hat_ref*).
As N grows the histogram should look increasingly Gaussian.

It uses IPOPT on the temperature box [340, 420] (single shooting) and warm-starts
every replicate from the reference solution (projected strictly interior), so the
many small solves converge quickly.  The study is run only for the RISK-NEUTRAL
SAA, so its output folder and files are named accordingly: the raw replicate values
are saved to output/risk-neutral-limit-theorem/risk-neutral_clt.json and the
histograms are rendered from them via ``ensemblecontrol.plot_clt``, so the figures
can be re-drawn without re-solving.

Usage (from this directory; prefix with MPLBACKEND=Agg on headless machines):
    ../../.venv/bin/python clt_batch_reactor.py
    ... --R 100 --n-ref 512          # cheaper run
"""

import argparse
import os
import sys

import numpy as np

import ensemblecontrol
from ensemblecontrol.inference import _BUILD_LOCK   # serialize CasADi construction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch_reactor import BatchReactor

HERE = os.path.dirname(os.path.abspath(__file__))
# CLT is run only on the RISK-NEUTRAL SAA value, so name the folder accordingly.
CLT_DIR = os.path.join(HERE, "output", "risk-neutral-limit-theorem")

STEPS = 10               # RK4 sub-steps per control interval (matches the SAA demo)
T_LO, T_HI = 340.0, 420.0


def strict_interior(w, lb=T_LO, ub=T_HI, margin=0.01):
    # Clip a warm start into the temperature box shrunk by a range-relative margin
    # so it is strictly feasible for IPOPT's interior-point method.
    span = ub - lb
    return np.clip(w, lb + margin * span, ub - margin * span)


def make_ipopt_solve(model, tol=1e-8):
    # A ready-made solve(samples, w0=None, inner_serial=False) -> (saa, w_opt, f_opt)
    # callback for clt_replication_study, using IPOPT (single shooting). Construction
    # is serialized under _BUILD_LOCK and the solve runs outside it, so it is safe
    # under the study's replicate threads; inner_serial pins the per-sample map serial
    # when those replicate solves are themselves threaded. The temperature box is far
    # from 0: the reference solve (w0=None) starts from a feasible interior profile
    # (380 K), and every warm-started replicate is projected strictly interior.
    guess = np.full((model.nintervals, model.ncontrols), 380.0)

    def solve(samples, w0=None, inner_serial=False):
        kw = dict(parallelization="serial", n_threads=1) if inner_serial else {}
        with _BUILD_LOCK:
            saa = ensemblecontrol.SAAProblem(model, samples, MultipleShooting=False,
                                             tol=tol, steps_per_interval=STEPS, **kw)
            if w0 is None:
                saa.initial_decisions = saa.initial_from_controls(guess)
            else:
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
                         "(default), 1, or an integer worker count")
    args = ap.parse_args()

    run_dir = CLT_DIR   # fixed, timestamp-free: output/limit_theorem/

    model = BatchReactor()
    # Reproducible i.i.d. scenario root (k20 ~ truncnorm(1000, 500, [500, 2000]));
    # clt_replication_study splits it into the reference stream plus one group per
    # sample size (each spawning R replicate streams).
    root = ensemblecontrol.TruncatedNormalSampler(
        1000.0, 500.0, 500.0, 2000.0, method="mc", seed=ROOT_SEED)
    # IPOPT on the temperature box [340, 420] (single shooting); each replicate is
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
        study["values_by_N"], os.path.join(run_dir, "risk-neutral_clt.json"),
        N_ref=study["n_ref"], f_ref=study["f_ref"], q=study["q"],
        meta={"model": "BatchReactor", "sampler": "TruncatedNormalSampler",
              "k20_distribution": "truncnorm(1000, 500, [500, 2000])",
              "seed": ROOT_SEED})
    # prefix "risk-neutral" -> figures risk-neutral_clt_N{N}.png / _clt_all.png
    ensemblecontrol.plot_clt(path, outdir=run_dir, stamp="", prefix="risk-neutral")


if __name__ == "__main__":
    main()
