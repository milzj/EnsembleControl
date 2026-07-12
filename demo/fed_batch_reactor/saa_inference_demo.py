"""Confidence intervals for the fed-batch reactor SAA optimal value.

Runs the two inference algorithms from ``ensemblecontrol.inference`` on the
fed-batch reactor:

  * plug-in CI (Algorithm 1)     -- normal interval under a unique optimizer,
  * subsampling CI (Algorithm 2) -- interval valid for nonunique optimizers.

Choose which to run with ``--algorithm {plugin,subsampling,both}`` (default
both).  The full-sample (size-N) SAA is solved once and reused: it is the
largest plug-in sample size AND the anchor J_hat_N* / u_hat_N for subsampling,
so running subsampling after the plug-in needs no extra size-N solve.

Each algorithm saves its raw data to JSON, then the figures are rendered from
those saved files -- so you can re-plot (rename labels, change CI bands) without
re-solving; see the plot-only path below.

Usage (from the repo root, module importable via the script's own directory):
    MPLBACKEND=Agg .venv/bin/python demo/fed_batch_reactor/saa_inference_demo.py
    ... --algorithm plugin
    ... --algorithm subsampling --m 200 --b 32
Re-plot only, no solves:
    .venv/bin/python -c "import ensemblecontrol as e; \
        e.plot_plugin('demo/fed_batch_reactor/output/inference/plugin.json', \
                      outdir='demo/fed_batch_reactor/output/inference')"
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ensemblecontrol
from fed_batch_reactor import FedBatchReactor

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "output", "inference")

SAMPLE_SIZES = (32, 64, 128)   # plug-in sweep; the last is the subsampling anchor
RADIUS = 0.2                   # relative parameter perturbation r
SEED = 0                       # scenario-sampler seed
OOS_SIZE = 512                 # fixed out-of-sample draw for the OOS-s.d. plug-in


def make_sampler():
    """Reproducible i.i.d. scenario sampler for the fed-batch parameters.

    A single seeded root is split with numpy stream-splitting into three
    independent streams: the SAA scenarios, the fixed (M = OOS_SIZE)
    out-of-sample draw, and the matched (M = N) out-of-sample draw.  All
    parameters are perturbed (frozen=()); the former constant parameter is now
    inlined in the model RHS.
    """
    nominal = FedBatchReactor().nominal_param[0]
    root = ensemblecontrol.UniformRelativeSampler(
        nominal, radius=RADIUS, method="mc", seed=SEED, frozen=())
    saa_stream, oos_fixed, oos_matched = root.spawn(3)
    return saa_stream, oos_fixed, oos_matched


def solve_saa(model, samples):
    saa = ensemblecontrol.SAAProblem(model, samples, MultipleShooting=False,
                                     tol=1e-8)
    w_opt, f_opt = saa.solve()
    return saa, w_opt, f_opt


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
    parser.add_argument("--seed-sub", type=int, default=1,
                        help="RNG seed for the subsample index draws")
    args = parser.parse_args()

    run_plugin = args.algorithm in ("plugin", "both")
    run_sub = args.algorithm in ("subsampling", "both")

    model = FedBatchReactor()
    saa_stream, oos_fixed, oos_matched = make_sampler()
    samples = saa_stream.sample(SAMPLE_SIZES[-1])
    os.makedirs(OUTDIR, exist_ok=True)
    N_full = SAMPLE_SIZES[-1]

    # One timestamped inference folder per run: output/inference/<stamp>/.
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = os.path.join(OUTDIR, stamp)

    # Solve each size-N SAA once (nested prefixes of the scenario draw). Reused
    # by BOTH algorithms: the plug-in sweep and, at each N, the subsampling
    # anchor J_hat_N* / u_hat_N -- so subsampling adds only the m subsample
    # re-solves per N, no extra size-N solves.
    solves = {N: solve_saa(model, samples[:N]) for N in SAMPLE_SIZES}

    plugin95 = oos95 = sub95 = None
    records = oos_records = oosN_records = sub_records = None
    plugin_path = plugin_oos_path = plugin_oosN_path = sub_path = None

    if run_plugin:
        # (i) plug-in CI with the IN-SAMPLE standard deviation (Algorithm 1)
        records = [ensemblecontrol.plugin_confidence_interval(
            solves[N][0], solves[N][1], f_opt=solves[N][2]) for N in SAMPLE_SIZES]
        plugin_path = os.path.join(run_dir, "plugin.json")
        ensemblecontrol.save_plugin_run(
            records, plugin_path,
            meta={"model": "FedBatchReactor", "sampler": "UniformRelativeSampler",
                  "radius": RADIUS, "seed": SEED})
        plugin95 = records[-1]["ci"]["levels"][0.95]

        # (ii) variant: standard deviation estimated OUT-OF-SAMPLE at the fixed
        # optimizer u_hat_N on a fresh independent draw; M = OOS_SIZE (fixed).
        oos_saa = ensemblecontrol.SAAProblem(
            model, oos_fixed.sample(OOS_SIZE), MultipleShooting=False, tol=1e-8)
        oos_records = [ensemblecontrol.plugin_oos_confidence_interval(
            solves[N][0], solves[N][1], oos_saa, f_opt=solves[N][2])
            for N in SAMPLE_SIZES]
        plugin_oos_path = os.path.join(run_dir, "plugin_oos.json")
        ensemblecontrol.save_plugin_run(
            oos_records, plugin_oos_path,
            meta={"model": "FedBatchReactor", "variance": "out-of-sample",
                  "oos_size": OOS_SIZE, "seed": SEED})
        oos95 = oos_records[-1]["ci"]["levels"][0.95]

        # (iii) same OOS variant but with M = N (matched to each training
        # size), on a separate independent stream.
        oosN_full = oos_matched.sample(N_full)
        oosN_records = [ensemblecontrol.plugin_oos_confidence_interval(
            solves[N][0], solves[N][1],
            ensemblecontrol.SAAProblem(model, oosN_full[:N],
                                       MultipleShooting=False, tol=1e-8),
            f_opt=solves[N][2]) for N in SAMPLE_SIZES]
        plugin_oosN_path = os.path.join(run_dir, "plugin_oos_matched.json")
        ensemblecontrol.save_plugin_run(
            oosN_records, plugin_oosN_path,
            meta={"model": "FedBatchReactor", "variance": "out-of-sample",
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
        sub_streams = np.random.SeedSequence(args.seed_sub).spawn(len(SAMPLE_SIZES))
        sub_records = []
        for N, ss in zip(SAMPLE_SIZES, sub_streams):
            saa, w_opt, f_opt = solves[N]
            sub_records.append(ensemblecontrol.subsampling_confidence_interval(
                saa, f_opt, b=b_of(N), m=m,
                rng=np.random.default_rng(ss), w_opt=w_opt))
        sub_path = os.path.join(run_dir, "subsampling.json")
        ensemblecontrol.save_subsampling_run(
            sub_records, sub_path,
            meta={"resolver": "scipy-lbfgsb-warmstart", "rng_seed": args.seed_sub})
        sub95 = sub_records[-1]["ci"]["levels"][0.95]

    # Render the figures (from the saved data). Share the optimal-value y-axis on
    # the _ci{level} interval plots across every CI family generated (in-sample
    # plug-in, both out-of-sample plug-in variants, subsampling) for comparison.
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
        ensemblecontrol.plot_plugin(plugin_path, outdir=run_dir, stamp=stamp,
                                    value_ylim=value_ylim)
        ensemblecontrol.plot_plugin(
            plugin_oos_path, outdir=run_dir, stamp=stamp, prefix="plugin-oos",
            value_ylim=value_ylim, labels=oos_labels)
        ensemblecontrol.plot_plugin(
            plugin_oosN_path, outdir=run_dir, stamp=stamp,
            prefix="plugin-oos-matched", value_ylim=value_ylim, labels=oos_labels)
    if run_sub:
        ensemblecontrol.plot_subsampling(sub_path, outdir=run_dir, stamp=stamp,
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
