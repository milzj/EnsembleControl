"""Batch reactor under parametric uncertainty (Terwiesch & Agarwal, 1995).

Reproduces the first numerical example of Terwiesch & Agarwal, "Robust input
policies for batch reactors under parametric uncertainty" (Chem. Eng. Commun.
131, 33-52), using the sample-average-approximation machinery of
``ensemblecontrol`` -- the same tools as the harmonic-oscillator demo.

The reactor runs 2A -> B -> C (B desired, C waste); the temperature profile T(t)
is optimized while the decomposition collision factor k20 is uncertain,
k20 ~ truncnorm(1000, 500) on [500, 2000].  The paper's three strategies map onto
the SAA risk settings:

  * nominal      -- solve at the mean k20 = E[k20] (~1000),
  * risk-neutral -- the paper's *robust* policy: min_u E[-B] over the ensemble,
  * CVaR         -- risk-averse; as beta -> 1 it approaches the paper's *minimax*
                    (worst-case) policy.

It then runs the two confidence-interval algorithms from
``ensemblecontrol.inference`` on the risk-neutral SAA optimal value J = E[-B]:

  * plug-in CI (Algorithm 1)     -- normal interval under a unique optimizer,
  * subsampling CI (Algorithm 2) -- interval valid for nonunique optimizers.

Choose which with ``--algorithm {plugin,subsampling,both}`` (default both).  Every
solve uses IPOPT (single shooting).  Control/state trajectories go to
output/controls-state/; the CI figures/JSON -- computed only for the RISK-NEUTRAL
problem -- go to output/risk-neutral-inference/ (files prefixed risk-neutral_...);
and a yield-vs-k20 comparison (the paper's Figure 2) plus a Table-1-style summary
are produced from the solved controls.

Usage (from this directory; prefix with MPLBACKEND=Agg on headless machines):
    ../../.venv/bin/python saa_batch_reactor.py
    ... --algorithm plugin
    ... --algorithm subsampling --m 200 --b 32
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import ensemblecontrol
from ensemblecontrol.inference import _BUILD_LOCK   # serialize CasADi construction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch_reactor import BatchReactor

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "output")
CS_DIR = os.path.join(OUTDIR, "controls-state")   # nominal/risk-neutral/CVaR plots
# Inference (plug-in / subsampling CI) is run only on the RISK-NEUTRAL SAA optimal
# value, so its folder and files are named accordingly.
CI_DIR = os.path.join(OUTDIR, "risk-neutral-inference")

SAMPLE_SIZES = (32, 64, 128)   # plug-in sweep; the last is the subsampling anchor
SEED = 0            # scenario-sampler seed
SUB_SEED = 1        # subsample index RNG seed
OOS_SIZE = 512      # fresh out-of-sample draw for the OOS-s.d. plug-in
STEPS = 10          # RK4 sub-steps per control interval (accurate rollout)
T_LO, T_HI = 340.0, 420.0       # temperature box (control bounds)
K20_LO, K20_HI = 500.0, 2000.0  # k20 uncertainty interval (Sec. 3.1)
LINESTYLES = ("-", "--", "-.")  # nominal / risk-neutral / CVaR -- distinct line types


def make_sampler():
    """i.i.d. Monte Carlo root for k20 ~ truncnorm(1000, 500) on [500, 2000],
    split into three independent streams: the SAA scenarios, the fixed
    (M = OOS_SIZE) out-of-sample draw, and the matched (M = N) out-of-sample draw
    (the last two feed the out-of-sample-s.d. plug-in CI)."""
    root = ensemblecontrol.TruncatedNormalSampler(
        1000.0, 500.0, K20_LO, K20_HI, method="mc", seed=SEED)
    saa_stream, oos_fixed, oos_matched = root.spawn(3)
    return saa_stream, oos_fixed, oos_matched


def solve_saa(model, samples, beta=0.0):
    # Single shooting + IPOPT via SAAProblem.solve(). beta=0 is risk-neutral (the
    # mean); 0 < beta < 1 selects the CVaR risk-averse objective. The temperature
    # box is far from 0, so warm-start from a feasible interior profile (mid-range
    # 380 K) rather than the default ~0 guess.
    saa = ensemblecontrol.SAAProblem(model, samples, beta=beta,
                                     MultipleShooting=False, tol=1e-8,
                                     steps_per_interval=STEPS)
    guess = np.full((model.nintervals, model.ncontrols), 380.0)
    saa.initial_decisions = saa.initial_from_controls(guess)
    w_opt, f_opt = saa.solve()
    return saa, w_opt, f_opt


def strict_interior(w, lb=T_LO, ub=T_HI, margin=0.01):
    # Clip a warm start into the temperature box shrunk by a range-relative margin
    # so it is strictly feasible for IPOPT's interior-point method -- controls that
    # saturate at a bound would otherwise start exactly on it.
    span = ub - lb
    return np.clip(w, lb + margin * span, ub - margin * span)


def ipopt_resolve_for(N, saa, w_opt):
    # Subsampling re-solver (Algorithm 2): solve each size-b subproblem with IPOPT,
    # warm-started -- strictly interior -- from the full-sample control. Build the
    # subproblem under _BUILD_LOCK with the inner per-sample map pinned serial, then
    # solve outside the lock, so the m re-solves can run in parallel (--workers)
    # without either racing on CasADi construction or oversubscribing the cores.
    controls = strict_interior(saa.control_matrix(w_opt))
    def resolve(indices):
        with _BUILD_LOCK:
            sub = saa.subproblem(indices, parallelization="serial", n_threads=1)
            sub.initial_decisions = sub.initial_from_controls(controls)
        return sub.solve()[1]
    return resolve


def plot_control(saa, w_opt, prefix, label):
    # Control trajectory T(t) -> output/controls-state/<prefix>_control.png.
    # ``label`` is the legend prefix; the CVaR solve passes a beta-annotated label
    # so the risk level shows on the plot.
    plotter = ensemblecontrol.SolutionPlotter(saa, w_opt)
    fig_c, ax_c = plotter.plot_controls(control_labels=[r"$T(t)$"], step=True,
                                        label_prefix=label)
    # Frame every control plot on the temperature box, with ticks delimited by
    # the bounds (340 .. 420 K) so all control plots share one y-axis.
    ax_c.set_ylim(T_LO, T_HI)
    ax_c.set_yticks(np.arange(T_LO, T_HI + 1.0, 20.0))
    ax_c.set_ylabel(r"$T$ [K]")
    fig_c.savefig(os.path.join(CS_DIR, prefix + "_control.png"))
    plt.close(fig_c)


def b_trajectory_stats(model, w_opt, samples):
    # Roll a fixed control across the k20 ensemble and return the time grid together
    # with the mean and standard deviation of the product [B] trajectory B(t, xi)
    # over the scenarios (same ensemble simulator as terminal_B, all time nodes).
    sim = ensemblecontrol.SAAProblem(model, samples, MultipleShooting=False,
                                     steps_per_interval=STEPS)
    traj = sim.ensemble_state_trajectory(sim.control_matrix(w_opt))
    B = traj[1::model.nstates, :]              # (nsamples, nintervals+1): B(t) / scenario
    tgrid = np.array([model.mesh_width * k for k in range(model.nintervals + 1)])
    return tgrid, B.mean(axis=0), B.std(axis=0)


def plot_states_B(prefix, label, tgrid, mean, std, ylim):
    # Product-only state plot: the [B] ensemble mean E[B(t, xi)] with a +/- 3 s.d.
    # band over the k20 draw -> output/controls-state/<prefix>_states.png. The
    # y-axis is shared across policies for a direct spread comparison.
    fig, ax = plt.subplots()
    ax.plot(tgrid, mean, color="C1", label=r"%s  $\mathbb{E}[B(t,\xi)]$" % label)
    ax.fill_between(tgrid, mean - 3.0 * std, mean + 3.0 * std, color="C1",
                    alpha=0.2, label=r"$\pm 3$ s.d.")
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$[B]$")
    ax.grid(True)
    ax.legend()
    fig.savefig(os.path.join(CS_DIR, prefix + "_states.png"))
    plt.close(fig)


def plot_all_controls(model, policies, savepath):
    # One figure overlaying every policy's temperature profile T(t), on the shared
    # [340, 420] box. Controls are piecewise constant (where='pre' holds control k
    # over (t_k, t_{k+1}]); the leading NaN matches len(tgrid).
    tgrid = np.array([model.mesh_width * k for k in range(model.nintervals + 1)])
    fig, ax = plt.subplots()
    for (label, w_opt), ls in zip(policies, LINESTYLES):
        u = np.asarray(w_opt, float).ravel()[:model.nintervals]   # front block = T(t)
        ax.step(tgrid, np.concatenate([[np.nan], u]), where="pre", ls=ls, label=label)
    ax.set_ylim(T_LO, T_HI)
    ax.set_yticks(np.arange(T_LO, T_HI + 1.0, 20.0))
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$T$ [K]")
    ax.grid(True)
    ax.legend()
    fig.savefig(savepath)
    plt.close(fig)


def terminal_B(model, w_opt, k20_values):
    """Terminal [B] (= x2(tf)) obtained by rolling a solved control across the
    given k20 scenarios, using the package's ensemble simulator."""
    k20_values = np.atleast_2d(np.asarray(k20_values, float).reshape(-1, 1))
    sim = ensemblecontrol.SAAProblem(model, k20_values, MultipleShooting=False,
                                     steps_per_interval=STEPS)
    controls = sim.control_matrix(w_opt)                 # (nintervals, ncontrols)
    traj = sim.ensemble_state_trajectory(controls)       # (nstates*nsamples, N+1)
    return traj[1::model.nstates, -1]                    # x2 terminal per scenario


def plot_yield_vs_k20(model, policies, savepath):
    # Paper Figure 2: terminal [B] as a function of k20 for each policy -- the
    # robustness comparison (a robust/CVaR profile trades peak yield near the
    # nominal k20 for a flatter, higher curve in the worst-case tail).
    k20_grid = np.linspace(K20_LO, K20_HI, 151)
    fig, ax = plt.subplots()
    for (label, w_opt), ls in zip(policies, LINESTYLES):
        ax.plot(k20_grid, terminal_B(model, w_opt, k20_grid), ls=ls, label=label)
    ax.axvline(1000.0, color="0.6", lw=0.8, ls=":")      # nominal k20
    ax.set_xlabel(r"$k_{20}$")
    ax.set_ylabel(r"$[B](t_f)$")
    ax.grid(True)
    ax.legend()
    fig.savefig(savepath)
    plt.close(fig)


def yield_summary(model, policies, samples):
    # Table-1 analog: final amount of B (maximized) at the nominal k20 and its
    # expectation over the ensemble, per policy.
    print("\n{:>16s}  {:>14s}  {:>14s}".format("policy", "B(k20=1000)", "E[B]"))
    for name, w_opt in policies:
        b_nom = float(terminal_B(model, w_opt, [1000.0])[0])
        e_b = float(np.mean(terminal_B(model, w_opt, samples)))
        print("{:>16s}  {:>14.6f}  {:>14.6f}".format(name, b_nom, e_b))


def risk_illustration(model, policies, savepath, n_oos=2000, seed=7):
    # WHY the risk ordering matters: draw a large INDEPENDENT out-of-sample set of
    # k20 ~ truncnorm, evaluate each policy's terminal yield [B] on it, and show the
    # yield distribution together with two decision metrics:
    #   * E[B]        -- the mean yield (risk-neutral maximizes this),
    #   * worst-5% B  -- the mean of the worst 5% of yields, i.e. -CVaR_0.95[-B]
    #                    (the risk-averse tail metric; CVaR maximizes this).
    # Risk-neutral beats nominal on E[B]; CVaR beats risk-neutral on worst-5% B.
    oos = ensemblecontrol.TruncatedNormalSampler(
        1000.0, 500.0, K20_LO, K20_HI, method="mc", seed=seed).sample(n_oos)
    fig, ax = plt.subplots()
    print("\n{:>16s}  {:>10s}  {:>12s}  {:>10s}".format(
        "policy", "E[B]", "worst-5% B", "min B"))
    for label, w_opt in policies:
        B = terminal_B(model, w_opt, oos)
        var5 = np.quantile(B, 0.05)                  # 5% Value-at-Risk (yield)
        cvar5 = float(B[B <= var5].mean())           # mean of the worst 5% yields
        _, _, patches = ax.hist(B, bins=50, histtype="step", density=True, lw=1.5,
                                label=label)
        color = patches[0].get_edgecolor()
        ax.axvline(float(B.mean()), color=color, ls="-", lw=1.0)   # mean
        ax.axvline(cvar5, color=color, ls="--", lw=1.0)            # worst-5% mean
        print("{:>16s}  {:>10.4f}  {:>12.4f}  {:>10.4f}".format(
            label, float(B.mean()), cvar5, float(B.min())))
    ax.set_xlabel(r"$[B](t_f)$")
    ax.set_ylabel(r"density over $k_{20}\sim$ truncnorm")
    ax.set_title(r"solid = mean $\mathbb{E}[B]$,  dashed = worst-5\% mean")
    ax.grid(True)
    ax.legend()
    fig.savefig(savepath)
    plt.close(fig)


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
    parser.add_argument("--workers", default="auto",
                        help="parallelism over the m subsampling re-solves per N: "
                             "'auto' (default), 1, or an integer count")
    args = parser.parse_args()

    run_plugin = args.algorithm in ("plugin", "both")
    run_sub = args.algorithm in ("subsampling", "both")

    model = BatchReactor()
    os.makedirs(CS_DIR, exist_ok=True)
    ci_dir = CI_DIR   # fixed, timestamp-free: output/inference/

    saa_stream, oos_fixed, oos_matched = make_sampler()

    # Full N-scenario ensemble, drawn once: it trains the risk-neutral/CVaR SAAs,
    # anchors the inference, and -- simulated under each solved control -- supplies
    # the +/- 3 s.d. B-state bands. mean() is analytic and draws nothing, so taking
    # the nominal scenario afterwards does not perturb this draw.
    samples = saa_stream.sample(SAMPLE_SIZES[-1])
    N_full = SAMPLE_SIZES[-1]

    # Nominal: single scenario at the mean k20 = E[k20] (~1000).
    saa_nom, w_nom, f_nom = solve_saa(model, np.atleast_2d(saa_stream.mean()))

    # Risk-neutral SAA over k20 ~ truncnorm -- the paper's "robust" policy. Solve
    # each size-N SAA once (nested prefixes); reused by BOTH CI algorithms -- the
    # plug-in sweep and, at each N, the subsampling anchor J_hat_N* / u_hat_N.
    solves = ensemblecontrol.solve_saa_prefixes(
        lambda s: solve_saa(model, s), samples, SAMPLE_SIZES)
    saa_rn, w_rn = solves[N_full][0], solves[N_full][1]

    # CVaR risk-averse on the SAME N-scenario ensemble (beta -> 1 approaches the
    # paper's minimax / worst case): beta = 0.95 optimizes the mean of the worst 5%
    # terminal-yield tail.
    beta = 0.95
    saa_cv, w_cv, f_cv = solve_saa(model, samples, beta=beta)
    cvar_label = r"CVaR $\beta={}$".format(beta)
    policies = [("nominal", w_nom), ("risk-neutral", w_rn), (cvar_label, w_cv)]

    # Per-policy control trajectories, then the product-only B-state ensembles.
    # Every control is simulated across the SAME full k20 ensemble, so the +/- 3
    # s.d. band is meaningful for every policy -- including the nominal one, whose
    # own solve carries a single scenario -- and directly comparable on a shared
    # y-axis (the nominal control, tuned to k20 = 1000, spreads most in the tail).
    plot_control(saa_nom, w_nom, "nominal", "nominal")
    plot_control(saa_rn, w_rn, "risk-neutral", "risk-neutral")
    plot_control(saa_cv, w_cv, "cvar-{:.2f}".format(beta), cvar_label)

    state_specs = [("nominal", "nominal", w_nom),
                   ("risk-neutral", "risk-neutral", w_rn),
                   ("cvar-{:.2f}".format(beta), cvar_label, w_cv)]
    stats = {prefix: b_trajectory_stats(model, w, samples)
             for prefix, _, w in state_specs}
    ymax = max(float((m + 3.0 * s).max()) for _, m, s in stats.values())
    for prefix, lab, _ in state_specs:
        tgrid, mean, std = stats[prefix]
        plot_states_B(prefix, lab, tgrid, mean, std, ylim=(0.0, 1.05 * ymax))

    # Paper Table 1 + Figure 2 from the solved controls; the all-policy control
    # overlay; and the out-of-sample yield distribution illustrating the risk
    # ordering (why risk-neutral > nominal in mean, CVaR > risk-neutral in the tail).
    yield_summary(model, policies, samples)
    plot_yield_vs_k20(model, policies, os.path.join(OUTDIR, "yield_vs_k20.png"))
    plot_all_controls(model, policies, os.path.join(CS_DIR, "all_controls.png"))
    risk_illustration(model, policies, os.path.join(OUTDIR, "yield_distribution.png"))

    plugin95 = oos95 = sub95 = None
    records = oos_records = oosN_records = sub_records = None
    plugin_path = plugin_oos_path = plugin_oosN_path = sub_path = None
    meta_base = {"model": "BatchReactor", "sampler": "TruncatedNormalSampler",
                 "k20_distribution": "truncnorm(1000, 500, [500, 2000])", "seed": SEED}

    if run_plugin:
        # (i) plug-in CI with the IN-SAMPLE standard deviation (Algorithm 1)
        records = ensemblecontrol.plugin_sweep(solves, SAMPLE_SIZES)
        plugin_path = os.path.join(ci_dir, "risk-neutral_plugin.json")
        ensemblecontrol.save_plugin_run(records, plugin_path, meta=meta_base)
        plugin95 = records[-1]["ci"]["levels"][0.95]

        # (ii) variant: standard deviation estimated OUT-OF-SAMPLE at the fixed
        # optimizer u_hat_N on a fresh independent draw (M = OOS_SIZE, fixed), so
        # sigma is not biased by the scenarios u_hat_N was fit to.
        oos_saa = ensemblecontrol.SAAProblem(
            model, oos_fixed.sample(OOS_SIZE), MultipleShooting=False, tol=1e-8,
            steps_per_interval=STEPS)
        oos_records = ensemblecontrol.plugin_oos_sweep(
            solves, SAMPLE_SIZES, lambda N: oos_saa)
        plugin_oos_path = os.path.join(ci_dir, "risk-neutral_plugin_oos.json")
        ensemblecontrol.save_plugin_run(
            oos_records, plugin_oos_path,
            meta={**meta_base, "variance": "out-of-sample", "oos_size": OOS_SIZE})
        oos95 = oos_records[-1]["ci"]["levels"][0.95]

        # (iii) same OOS variant but with M = N (out-of-sample size matched to each
        # training size), on a separate independent stream.
        oosN_full = oos_matched.sample(N_full)
        oosN_records = ensemblecontrol.plugin_oos_sweep(
            solves, SAMPLE_SIZES,
            lambda N: ensemblecontrol.SAAProblem(
                model, oosN_full[:N], MultipleShooting=False, tol=1e-8,
                steps_per_interval=STEPS))
        plugin_oosN_path = os.path.join(ci_dir, "risk-neutral_plugin_oos_matched.json")
        ensemblecontrol.save_plugin_run(
            oosN_records, plugin_oosN_path,
            meta={**meta_base, "variance": "out-of-sample", "oos_size": "N"})

    if run_sub:
        # subsampling at each sample size N -- the analogue of the plug-in sweep.
        # Default block size b = floor(N^{6/7}) PER N; m = 5*max(N) subsamples,
        # constant. --b/--m override; --workers threads the m IPOPT re-solves.
        b_of = ((lambda N: args.b) if args.b is not None
                else ensemblecontrol.default_subsample_size)
        m = (args.m if args.m is not None
             else ensemblecontrol.default_num_subsamples(SAMPLE_SIZES[-1]))
        try:
            sub_records = ensemblecontrol.subsampling_sweep(
                solves, SAMPLE_SIZES, b_of=b_of, m=m, seed=SUB_SEED,
                resolve_for=ipopt_resolve_for, workers=args.workers, progress=True)
        except ValueError as err:
            parser.error(str(err))
        sub_path = os.path.join(ci_dir, "risk-neutral_subsampling.json")
        ensemblecontrol.save_subsampling_run(
            sub_records, sub_path,
            meta={"resolver": "ipopt-warmstart", "rng_seed": SUB_SEED})
        sub95 = sub_records[-1]["ci"]["levels"][0.95]

    # Render the CI figures from the saved data, sharing the optimal-value y-axis
    # across every family generated for direct comparison.
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
    if run_plugin:
        ensemblecontrol.plot_plugin(plugin_path, outdir=ci_dir, stamp="",
                                    prefix="risk-neutral_plugin", value_ylim=value_ylim)
        ensemblecontrol.plot_plugin(plugin_oos_path, outdir=ci_dir, stamp="",
                                    prefix="risk-neutral_plugin-oos", value_ylim=value_ylim)
        ensemblecontrol.plot_plugin(plugin_oosN_path, outdir=ci_dir, stamp="",
                                    prefix="risk-neutral_plugin-oos-matched",
                                    value_ylim=value_ylim)
    if run_sub:
        ensemblecontrol.plot_subsampling(sub_path, outdir=ci_dir, stamp="",
                                         prefix="risk-neutral_subsampling",
                                         value_ylim=value_ylim)

    # The SAA optimal value J = E[-B] is negative (a minimization of -B); its CI is
    # reported below (subtract from 0 to read it as a bound on the expected yield).
    print("\n95% confidence interval for J_hat_N* = E[-B] (N = {}):".format(N_full))
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
