"""Threaded, per-sample single-shooting transcription via CasADi Function.map.

SingleShootingProblem integrates the whole ensemble as one stacked
(nstates*nsamples) symbolic graph, so Ipopt's objective/gradient evaluation runs
single-threaded.  MappedSingleShootingProblem produces the *same* NLP -- identical
decision layout [U_0..U_{N-1}, <CVaR/TV tail>] and objective
(int L dt + (1/nsamples) sum_i F(x_i(T))) -- but evaluates the samples in parallel:

    rollout.map("rollout_map", "thread", nsamples,
                reduce_in=["U_all"], reduce_out=["qf"])

- reduce_in=["U_all"]: the single control sequence is shared across all samples.
- reduce_out=["qf"]: the per-sample integral cost is summed.
- per-sample terminal states come back as an (nstates x nsamples) matrix.

Two effects compound: the "thread" backend spreads the samples over cores, and
expand() (MX -> flat SX) before mapping removes the per-interval Function-call
overhead that would otherwise make the mapped graph ~16x more expensive to walk.
Both are required for the speedup.

Both RK4 and explicit Euler are supported (steps_per_interval sub-steps per
control interval; see mapped_common.build_interval).  "openmp" silently falls
back to serial on the pip CasADi wheel, so use "thread".
"""

import os

import numpy as np
from casadi import *

from .mapped_common import _samples_2d, build_interval
from .risk_measures import risk_measure
from .total_variation import tv_regularization


def build_rollout_map(control_problem, samples, parallelization="thread",
                      n_threads=None, steps_per_interval=4, expand=True,
                      scheme="rk4"):
    """Per-sample full-horizon rollout, mapped over the ensemble.

    Inputs : x0 (nstates x nsamples), U_all (ncontrols x nintervals, shared),
             k (nparams x nsamples).
    Outputs: xf (nstates x nsamples), qf (1 x 1, integral cost summed over samples).
    """
    cp = control_problem
    samples = _samples_2d(samples)
    nstates = cp.nstates
    ncontrols = cp.ncontrols
    nintervals = cp.nintervals
    nsamples = samples.shape[0]
    nparams = samples.shape[1]

    # one interval for a single sample (steps_per_interval sub-steps of `scheme`)
    interval = build_interval(cp, nparams, steps_per_interval, scheme)

    # --- full horizon for a single sample (composes the interval integrator) ---
    X0 = MX.sym("X0", nstates)
    U_all = MX.sym("U_all", ncontrols, nintervals)
    K = MX.sym("K", nparams)
    X = X0
    Q = 0.0
    for j in range(nintervals):
        r = interval(x0=X, u=U_all[:, j], k=K)
        X = r["xf"]
        Q = Q + r["qf"]
    rollout = Function("rollout", [X0, U_all, K], [X, Q],
                       ["x0", "U_all", "k"], ["xf", "qf"])

    # Flatten to a scalar (SX) function before mapping: without this the mapped
    # rollout is walked as nintervals MX Function-call nodes per sample, inflating
    # total CPU ~16x and swamping the parallel gain.  With expand each thread does
    # only the arithmetic.
    if expand:
        rollout = rollout.expand()

    if n_threads is None:
        n_threads = min(nsamples, os.cpu_count() or 1)

    # map over samples: U_all shared (reduce_in), integral cost summed (reduce_out)
    return rollout.map("rollout_map", parallelization, nsamples,
                       ["U_all"], ["qf"], {"max_num_threads": int(n_threads)})


def MappedSingleShootingProblem(control_problem,
                                samples,
                                control_bounds,
                                nsamples,
                                nintervals,
                                beta=0.0,
                                tv_rho=0.0,
                                parallelization="thread",
                                n_threads=None,
                                expand=True,
                                steps_per_interval=4,
                                scheme="rk4",
                                terminal_constraints=None):
    """Same NLP as SingleShootingProblem, ensemble evaluated with a threaded map.

    Signature mirrors SingleShootingProblem but takes control_problem + samples
    (needed to build the per-sample rollout) instead of the stacked dynamics.

    terminal_constraints: optional list of (sample, func, lb, ub) enforcing
    lb <= func(x^(sample)(t_f)) <= ub on individual ensemble members, read
    directly off the per-sample terminal-state matrix.
    """
    cp = control_problem
    nstates = cp.nstates
    ncontrols = cp.ncontrols
    samples = _samples_2d(samples)

    lbu, ubu = control_bounds[0], control_bounds[1]
    u0 = [(a + b) / 2 for a, b in zip(lbu, ubu)]
    u0 = list(np.nan_to_num(u0, posinf=0.0, neginf=0.0))

    rollout_map = build_rollout_map(cp, samples, parallelization=parallelization,
                                    n_threads=n_threads, expand=expand,
                                    steps_per_interval=steps_per_interval,
                                    scheme=scheme)

    # per-sample constant data, sample-contiguous to match risk_measure indexing
    X0 = np.array([cp.parameterized_initial_state(samples[i])
                   for i in range(nsamples)], dtype=float).T   # nstates x nsamples
    P = samples.T                                              # nparams x nsamples

    # decision vector: one control per interval, exactly like SingleShootingProblem
    w, w0, lbw, ubw = [], [], [], []
    g, lbg, ubg = [], [], []
    controls = []
    for k in range(nintervals):
        Uk = MX.sym("U_" + str(k), ncontrols)
        w += [Uk]
        lbw += lbu
        ubw += ubu
        w0 += u0
        controls += [Uk]

    U_all = horzcat(*controls)                                # ncontrols x nintervals

    res = rollout_map(x0=DM(X0), U_all=U_all, k=DM(P))
    Xk_mat = res["xf"]                                        # nstates x nsamples
    J = res["qf"] / nsamples                                  # average integral cost

    # column-major flatten -> [x_0(T); x_1(T); ...], the layout risk_measure indexes
    Xk = vec(Xk_mat)

    # nominal terminal state for the CVaR warm-start (only read when beta>0): roll
    # the mean control through the same per-sample dynamics.
    if 0.0 < beta < 1.0:
        U0_all = DM(np.tile(np.reshape(u0, (ncontrols, 1)), (1, nintervals)))
        xk = np.array(rollout_map(x0=DM(X0), U_all=U0_all, k=DM(P))["xf"]).T.flatten()
    else:
        xk = np.zeros(nstates * nsamples)

    J, w, lbw, ubw, g, lbg, ubg = risk_measure(
        J, w, w0, lbw, ubw, g, lbg, ubg, cp.final_cost_function,
        Xk, xk, beta, nstates * nsamples, nsamples)
    J, w, lbw, ubw, g, lbg, ubg = tv_regularization(
        J, w, w0, lbw, ubw, g, lbg, ubg, controls, tv_rho, ncontrols)

    # deterministic terminal-state constraints on chosen ensemble members: read
    # the sample's terminal state straight off the per-sample rollout output.
    for sample, func, lb, ub in (terminal_constraints or []):
        expr = func(Xk_mat[:, sample])
        n = expr.numel()
        g += [expr]
        lbg += list(np.broadcast_to(np.asarray(lb, dtype=float), n))
        ubg += list(np.broadcast_to(np.asarray(ub, dtype=float), n))

    objective = J
    constraints = vertcat(*g)
    decisions = vertcat(*w)

    # Diagonal of the discretized-L2 mass matrix over the decision vector: the
    # control block (leading ncontrols*nintervals entries) carries the mesh width
    # h; the CVaR/TV slack tail is not a discretized function so it stays 1.
    metric_weights = np.ones(decisions.numel())
    metric_weights[:ncontrols * nintervals] = cp.mesh_width

    return objective, constraints, decisions, w0, [lbw, ubw, lbg, ubg], \
        metric_weights
