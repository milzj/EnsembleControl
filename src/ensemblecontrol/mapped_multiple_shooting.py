"""Threaded, per-sample multiple-shooting transcription via CasADi Function.map.

The serial MultipleShootingProblem integrates the whole ensemble as one stacked
(nstates*nsamples) symbolic graph, so Ipopt's objective/gradient/Jacobian runs
single-threaded.  MappedMultipleShootingProblem produces the *same* NLP --
identical decision layout

    [X0, U0, X1, U1, ..., U_{N-1}, X_N, <CVaR/TV tail>]

with every X_k the full stacked ensemble state (sample-contiguous), one control
per interval, and per-interval continuity gaps -- but evaluates the shooting
intervals in parallel.

Unlike single shooting, the shooting nodes are decision variables, so there is
no full-horizon composition: every (interval, sample) integration is
independent.  We therefore map ONE per-sample interval integrator over the whole
nintervals*nsamples grid in a single Function.map call (one thread dispatch):

    interval.map("ms_interval_map", "thread", nintervals*nsamples)

Inputs are laid out interval-major (column k*nsamples + i is interval k,
sample i):
- x0 : each shooting node's per-sample state (from the state decision vars),
- u  : the interval's control, repeated across that interval's samples,
- k  : the per-sample parameter.
The mapped terminal states xf feed the continuity residuals vec(xf_k) - X_{k+1},
and the summed running cost gives J = sum(qf)/nsamples.

expand() (MX -> flat SX) before mapping removes the per-sub-step Function-call
overhead so each thread does only arithmetic; both it and the "thread" backend
are needed for the speedup.  "openmp" silently degrades to serial on the pip
CasADi wheel, so use "thread".
"""

import os

import numpy as np
from casadi import *

from .mapped_common import _samples_2d, build_interval
from .risk_measures import risk_measure
from .total_variation import tv_regularization


def MappedMultipleShootingProblem(control_problem,
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
    """Same NLP as MultipleShootingProblem, ensemble evaluated with a threaded map.

    Signature mirrors MappedSingleShootingProblem: takes control_problem +
    samples (to build the per-sample interval integrator) instead of a stacked
    dynamics function.

    terminal_constraints: optional list of (sample, func, lb, ub) enforcing
    lb <= func(x^(sample)(t_f)) <= ub on individual ensemble members, read off
    the final shooting node.
    """
    cp = control_problem
    nstates = cp.nstates                       # per-sample state dimension
    ncontrols = cp.ncontrols
    samples = _samples_2d(samples)
    nparams = samples.shape[1]
    nstates_total = nstates * nsamples

    lbu, ubu = control_bounds[0], control_bounds[1]
    u0 = [(a + b) / 2 for a, b in zip(lbu, ubu)]
    u0 = list(np.nan_to_num(u0, posinf=0.0, neginf=0.0))

    # per-sample single interval, flattened to SX so each mapped thread only does
    # arithmetic (see module docstring)
    interval = build_interval(cp, nparams, steps_per_interval, scheme)
    if expand:
        interval = interval.expand()

    # per-sample constant data, columns = samples (sample-contiguous layout)
    X0_data = np.array([cp.parameterized_initial_state(samples[i])
                        for i in range(nsamples)], dtype=float).T   # nstates x nsamples
    P = samples.T                                                   # nparams x nsamples

    # --- numeric nominal rollout with the mean control, for the warm start ---
    # Roll all samples one interval at a time with u0 (shared control) to seed
    # each shooting node -- exactly the serial MultipleShootingProblem warm start.
    nominal_map = interval.map("ms_nominal_map", "serial", nsamples,
                               ["u"], [], {"max_num_threads": 1})
    node_states = [X0_data]                                         # per node: nstates x nsamples
    xk = X0_data
    for _ in range(nintervals):
        xk = np.array(nominal_map(x0=DM(xk), u=DM(u0), k=DM(P))["xf"], dtype=float)
        node_states.append(xk.reshape(nstates, nsamples))

    # --- decision vector: lifted states per node + one control per interval ---
    w, w0, lbw, ubw = [], [], [], []
    g, lbg, ubg = [], [], []
    controls = []
    state_nodes = []

    # Diagonal of the discretized-L2 mass matrix, built in lockstep with w: only
    # the control DOFs are discretized functions and carry the mesh width h; the
    # lifted shooting states (and the CVaR/TV slack tail appended later) stay 1.
    sc_weights = []

    init = list(node_states[0].T.flatten())                        # X0 sample-contiguous
    X0sym = MX.sym("X0", nstates_total)
    w += [X0sym]
    lbw += init
    ubw += init
    w0 += init
    state_nodes += [X0sym]
    sc_weights += nstates_total * [1.0]

    for k in range(nintervals):
        Uk = MX.sym("U_" + str(k), ncontrols)
        w += [Uk]
        lbw += lbu
        ubw += ubu
        w0 += u0
        controls += [Uk]
        sc_weights += ncontrols * [cp.mesh_width]

        Xk1 = MX.sym("X_" + str(k + 1), nstates_total)
        w += [Xk1]
        lbw += nstates_total * [-inf]
        ubw += nstates_total * [inf]
        w0 += list(node_states[k + 1].T.flatten())
        state_nodes += [Xk1]
        sc_weights += nstates_total * [1.0]

    # --- one mapped call over the whole (interval, sample) grid ---------------
    M = nintervals * nsamples
    if n_threads is None:
        n_threads = min(M, os.cpu_count() or 1)
    interval_map = interval.map("ms_interval_map", parallelization, M,
                                [], [], {"max_num_threads": int(n_threads)})

    # interval-major columns: k*nsamples + i is interval k, sample i
    x0_mat = horzcat(*[reshape(state_nodes[k], nstates, nsamples)
                       for k in range(nintervals)])                 # nstates x M
    u_mat = horzcat(*[repmat(controls[k], 1, nsamples)
                      for k in range(nintervals)])                  # ncontrols x M
    k_mat = np.tile(P, (1, nintervals))                            # nparams x M

    res = interval_map(x0=x0_mat, u=u_mat, k=DM(k_mat))
    xf_mat = res["xf"]                                             # nstates x M
    J = sum2(res["qf"]) / nsamples                                # average integral cost

    # continuity gaps: predicted end of interval k must equal shooting node k+1
    for k in range(nintervals):
        xf_block = xf_mat[:, k * nsamples:(k + 1) * nsamples]      # nstates x nsamples
        g += [vec(xf_block) - state_nodes[k + 1]]                 # both sample-contiguous
        lbg += nstates_total * [0.0]
        ubg += nstates_total * [0.0]

    # terminal cost / CVaR on the final shooting node
    Xk = state_nodes[-1]
    # numeric nominal terminal state (only read by the CVaR warm start)
    xk_final = node_states[-1].T.flatten()
    J, w, lbw, ubw, g, lbg, ubg = risk_measure(
        J, w, w0, lbw, ubw, g, lbg, ubg, cp.final_cost_function,
        Xk, xk_final, beta, nstates_total, nsamples)
    J, w, lbw, ubw, g, lbg, ubg = tv_regularization(
        J, w, w0, lbw, ubw, g, lbg, ubg, controls, tv_rho, ncontrols)

    # deterministic terminal-state constraints on chosen ensemble members
    Xk_final_mat = reshape(state_nodes[-1], nstates, nsamples)    # column i = sample i
    for sample, func, lb, ub in (terminal_constraints or []):
        expr = func(Xk_final_mat[:, sample])
        n = expr.numel()
        g += [expr]
        lbg += list(np.broadcast_to(np.asarray(lb, dtype=float), n))
        ubg += list(np.broadcast_to(np.asarray(ub, dtype=float), n))

    objective = J
    constraints = vertcat(*g)
    decisions = vertcat(*w)

    # slack tail (CVaR/TV) is appended after the state/control block -> weight 1
    metric_weights = np.ones(decisions.numel())
    metric_weights[:len(sc_weights)] = sc_weights

    return objective, constraints, decisions, w0, [lbw, ubw, lbg, ubg], \
        metric_weights
