"""Harmonic oscillator under parametric uncertainty (arXiv:2407.18182).

Solves the risk-neutral sample-average approximation of the harmonic-oscillator
control problem of Melnikov & Milz -- uncertain angular frequency
k ~ U[0, 2*pi] -- and then runs the two confidence-interval algorithms from
``ensemblecontrol.inference`` on the SAA optimal value:

  * plug-in CI (Algorithm 1)     -- normal interval under a unique optimizer,
  * subsampling CI (Algorithm 2) -- interval valid for nonunique optimizers.

Choose which to run with ``--algorithm {plugin,subsampling,both}`` (default
both).  The full-sample (size-N) SAA is solved once and reused as the largest
plug-in sample size and as the subsampling anchor J_hat_N* / u_hat_N, so running
subsampling after the plug-in needs no extra size-N solve.  Each algorithm saves
its raw data to JSON and the figures are rendered from those files, so they can
be re-plotted (relabelled, re-banded) without re-solving.

The statistical procedures assume i.i.d. scenarios, so k is drawn by i.i.d.
Monte Carlo (UniformSampler, method="mc").  Single shooting is used so the
subsampling solves warm-start from the full-sample control with scipy L-BFGS-B.

Usage (from this directory; prefix with MPLBACKEND=Agg on headless machines):
    ../../.venv/bin/python saa_harmonic_oscillator.py
    ... --algorithm plugin
    ... --algorithm subsampling --m 200 --b 32
Re-plot only (no solves):
    ../../.venv/bin/python -c "import ensemblecontrol as e; \
        e.plot_plugin('output/inference/plugin.json', outdir='output/inference')"
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import ensemblecontrol

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harmonic_oscillator import HarmonicOscillator

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "output")
CI_DIR = os.path.join(OUTDIR, "inference")

SAMPLE_SIZES = (32, 64, 128)   # plug-in sweep; the last is the subsampling anchor
SEED = 0                       # scenario-sampler seed
SUB_SEED = 1                   # subsample index RNG seed
OOS_SIZE = 512                 # fresh out-of-sample draw for the OOS-s.d. plug-in


def make_sampler():
    """i.i.d. Monte Carlo sampler for the uncertain frequency k ~ U[0, 2*pi].

    A single seeded root is split with numpy stream-splitting into two independent
    streams: the first drives the SAA scenarios, the second the independent
    out-of-sample draw used for the out-of-sample-s.d. plug-in CI.  i.i.d.
    (method="mc") is required for the statistical procedures.
    """
    root = ensemblecontrol.UniformSampler(0.0, 2.0 * np.pi, method="mc", seed=SEED)
    # independent streams: SAA scenarios, the fixed (N_out = OOS_SIZE) out-of-
    # sample draw, and the matched (N_out = N) out-of-sample draw.
    saa_stream, oos_fixed, oos_matched = root.spawn(3)
    return saa_stream, oos_fixed, oos_matched


def solve_saa(model, samples):
    # Single shooting + control box [-3, 3] so the subsampling solves can
    # warm-start with scipy L-BFGS-B from the full-sample control.
    saa = ensemblecontrol.SAAProblem(model, samples, MultipleShooting=False,
                                     tol=1e-8)
    w_opt, f_opt = saa.solve()
    return saa, w_opt, f_opt


def plot_solution(saa, w_opt, prefix):
    plotter = ensemblecontrol.SolutionPlotter(saa, w_opt)
    plotter.plot_controls(control_labels=[r"$u_1^*(t)$", r"$u_2^*(t)$"],
                          label_prefix=prefix,
                          savepath=os.path.join(OUTDIR, prefix + "_controls.png"))
    plotter.plot_states(
        state_labels=[r"$\mathbb{E}[x_1^*(t,\xi)]$",
                      r"$\mathbb{E}[x_2^*(t,\xi)]$"],
        label_prefix=prefix,
        savepath=os.path.join(OUTDIR, prefix + "_states.png"))
    plt.close("all")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("plugin", "subsampling", "both"),
                        default="both", help="which CI algorithm(s) to run")
    parser.add_argument("--m", type=int, default=None,
                        help="number of subsamples (Algorithm 2); "
                             "default 5*max(N), constant across the sweep")
    parser.add_argument("--b", type=int, default=None,
                        help="subsample size (Algorithm 2); default "
                             "floor(N^(6/7)) per sample size; must be < N")
    args = parser.parse_args()

    run_plugin = args.algorithm in ("plugin", "both")
    run_sub = args.algorithm in ("subsampling", "both")

    model = HarmonicOscillator()
    os.makedirs(OUTDIR, exist_ok=True)

    # One timestamped inference folder per run: output/inference/<stamp>/.
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    ci_dir = os.path.join(CI_DIR, stamp)

    saa_stream, oos_fixed, oos_matched = make_sampler()

    # Nominal problem: replace the random parameter by its expectation E[xi]
    # (Melnikov & Milz, arXiv:2407.18182). For k ~ U[0, 2*pi] this is k = pi;
    # using k = 0 would decouple the oscillator into two integrators and give the
    # wrong (non-oscillatory) nominal control.
    saa_nom, w_nom, f_nom = solve_saa(model, np.atleast_2d(saa_stream.mean()))
    plot_solution(saa_nom, w_nom, "nominal")

    # Risk-neutral SAA over the uncertain ensemble (k ~ U[0, 2*pi]). Solve each
    # size-N SAA once (nested prefixes); reused by BOTH algorithms -- the plug-in
    # sweep and, at each N, the subsampling anchor J_hat_N* / u_hat_N.
    samples = saa_stream.sample(SAMPLE_SIZES[-1])
    N_full = SAMPLE_SIZES[-1]
    solves = {N: solve_saa(model, samples[:N]) for N in SAMPLE_SIZES}
    plot_solution(solves[N_full][0], solves[N_full][1], "risk-neutral")

    print("nominal      objective  J = {:.8e}".format(float(np.ravel(f_nom)[0])))
    print("risk-neutral objective  J = {:.8e}".format(
        float(np.ravel(solves[N_full][2])[0])))

    plugin95 = oos95 = sub95 = None
    records = oos_records = oosN_records = sub_records = None
    plugin_path = plugin_oos_path = plugin_oosN_path = sub_path = None

    if run_plugin:
        # (i) plug-in CI with the IN-SAMPLE standard deviation (Algorithm 1)
        records = [ensemblecontrol.plugin_confidence_interval(
            solves[N][0], solves[N][1], f_opt=solves[N][2]) for N in SAMPLE_SIZES]
        plugin_path = os.path.join(ci_dir, "plugin.json")
        ensemblecontrol.save_plugin_run(
            records, plugin_path,
            meta={"model": "HarmonicOscillator", "sampler": "UniformSampler",
                  "k_distribution": "U[0, 2*pi]", "seed": SEED})
        plugin95 = records[-1]["ci"]["levels"][0.95]

        # (ii) variant: standard deviation estimated OUT-OF-SAMPLE at the fixed
        # optimizer u_hat_N on a fresh independent draw, so sigma is not biased by
        # the scenarios u_hat_N was fit to.  N_out = OOS_SIZE (large, fixed).
        oos_saa = ensemblecontrol.SAAProblem(
            model, oos_fixed.sample(OOS_SIZE), MultipleShooting=False, tol=1e-8)
        oos_records = [ensemblecontrol.plugin_oos_confidence_interval(
            solves[N][0], solves[N][1], oos_saa, f_opt=solves[N][2])
            for N in SAMPLE_SIZES]
        plugin_oos_path = os.path.join(ci_dir, "plugin_oos.json")
        ensemblecontrol.save_plugin_run(
            oos_records, plugin_oos_path,
            meta={"model": "HarmonicOscillator", "variance": "out-of-sample",
                  "oos_size": OOS_SIZE, "seed": SEED})
        oos95 = oos_records[-1]["ci"]["levels"][0.95]

        # (iii) same OOS variant but with N_out = N (out-of-sample size matched to
        # each training size), on a separate independent stream.
        oosN_full = oos_matched.sample(N_full)
        oosN_records = [ensemblecontrol.plugin_oos_confidence_interval(
            solves[N][0], solves[N][1],
            ensemblecontrol.SAAProblem(model, oosN_full[:N],
                                       MultipleShooting=False, tol=1e-8),
            f_opt=solves[N][2]) for N in SAMPLE_SIZES]
        plugin_oosN_path = os.path.join(ci_dir, "plugin_oos_matched.json")
        ensemblecontrol.save_plugin_run(
            oosN_records, plugin_oosN_path,
            meta={"model": "HarmonicOscillator", "variance": "out-of-sample",
                  "oos_size": "N", "seed": SEED})

    if run_sub:
        # subsampling at each sample size N -- the analogue of the plug-in sweep.
        # Default subsample size b = floor(N^{6/7}) PER sample size (grows with N,
        # b/N -> 0). It is not held constant at the largest-N value because
        # floor(Nmax^{6/7}) exceeds the smallest N. m = 5*Nmax subsamples, constant.
        # --b/--m override with a fixed value; the legend prints the numbers used.
        Nmax = SAMPLE_SIZES[-1]

        def b_of(N):  # +1e-9 guards the float rounding at exact powers (128^{6/7}=64)
            return args.b if args.b is not None else int(np.floor(N ** (6.0 / 7.0) + 1e-9))

        m = args.m if args.m is not None else 5 * Nmax
        for N in SAMPLE_SIZES:
            if not (0 < b_of(N) < N):
                parser.error("subsample size b = {} must satisfy 0 < b < N = {}"
                             .format(b_of(N), N))
        # Independent subsample-index streams per N (spawned for independence).
        sub_streams = np.random.SeedSequence(SUB_SEED).spawn(len(SAMPLE_SIZES))
        sub_records = []
        for N, ss in zip(SAMPLE_SIZES, sub_streams):
            saa, w_opt, f_opt = solves[N]
            sub_records.append(ensemblecontrol.subsampling_confidence_interval(
                saa, f_opt, b=b_of(N), m=m,
                rng=np.random.default_rng(ss), w_opt=w_opt))
        sub_path = os.path.join(ci_dir, "subsampling.json")
        ensemblecontrol.save_subsampling_run(
            sub_records, sub_path,
            meta={"resolver": "scipy-lbfgsb-warmstart", "rng_seed": SUB_SEED})
        sub95 = sub_records[-1]["ci"]["levels"][0.95]

    # Render the figures (from the saved data). Share the optimal-value y-axis on
    # the _ci{level} interval plots across every CI family generated (in-sample
    # plug-in, out-of-sample plug-in, subsampling) for direct comparison.
    ci_groups = []
    if run_plugin:
        ci_groups += [[r["ci"] for r in records], [r["ci"] for r in oos_records],
                      [r["ci"] for r in oosN_records]]
    if run_sub:
        ci_groups += [[r["ci"] for r in sub_records]]
    value_ylim = None
    if len(ci_groups) >= 2:
        value_ylim = ensemblecontrol.value_ylim_across(
            [ci for group in ci_groups for ci in group])
    oos_labels = {"loss_label": r"$F(x_{\widehat{u}_N}(t_f,\xi'_j))$"}
    if run_plugin:
        ensemblecontrol.plot_plugin(plugin_path, outdir=ci_dir, stamp=stamp,
                                    value_ylim=value_ylim)
        ensemblecontrol.plot_plugin(
            plugin_oos_path, outdir=ci_dir, stamp=stamp, prefix="plugin-oos",
            value_ylim=value_ylim, labels=oos_labels)
        ensemblecontrol.plot_plugin(
            plugin_oosN_path, outdir=ci_dir, stamp=stamp,
            prefix="plugin-oos-matched", value_ylim=value_ylim, labels=oos_labels)
    if run_sub:
        ensemblecontrol.plot_subsampling(sub_path, outdir=ci_dir, stamp=stamp,
                                         value_ylim=value_ylim)

    print("\n95% confidence interval for J_hat_N* (N = {}):".format(N_full))
    if plugin95 is not None:
        print("  plug-in (in-sample s.d.) : [{:.6e}, {:.6e}]  half-width {:.6e}"
              .format(plugin95["lo"], plugin95["hi"], plugin95["halfwidth"]))
    if oos95 is not None:
        print("  plug-in (out-of-sample)  : [{:.6e}, {:.6e}]  half-width {:.6e}"
              .format(oos95["lo"], oos95["hi"], oos95["halfwidth"]))
    if sub95 is not None:
        print("  subsampling              : [{:.6e}, {:.6e}]  half-width {:.6e}"
              .format(sub95["lo"], sub95["hi"], (sub95["hi"] - sub95["lo"]) / 2.0))


if __name__ == "__main__":
    main()
