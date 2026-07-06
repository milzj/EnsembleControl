"""Correctness of the threaded, per-sample single-shooting map transcription.

The serial stacked transcription that used to serve as the reference oracle has
been removed, so these tests validate the map against oracles that do not depend
on it: an independent numpy RK4 rollout of the DoubleIntegrator SAA objective,
finite-difference gradients, and equivalence of the "thread" and "serial" map
backends.
"""
import numpy as np
import ensemblecontrol

from .double_integrator import DoubleIntegrator

# Quiet Ipopt during the test run.
_IPOPT = {"print_level": 0, "sb": "yes"}

# Distinct parameter samples exercise the map over the ensemble (nsamples > 1).
# DoubleIntegrator's dynamics ignore the parameter, so the ensemble is degenerate
# -- fine here: these tests exercise the map assembly, not parameter spread.
_SAMPLES = [[0.0], [0.1], [-0.2]]
_NINTERVALS = 20


def _scalar(x):
    return float(np.asarray(x).ravel()[0])


def _build(samples=_SAMPLES, beta=0.0, tv_rho=0.0, integrator="rk4",
           parallelization="serial"):
    # Correctness tests use the "serial" map backend: it exercises the full
    # expand()/map assembly without paying per-evaluation thread-spawn overhead.
    # A dedicated test below runs the production "thread" backend.
    model = DoubleIntegrator()
    model.nintervals = _NINTERVALS
    return ensemblecontrol.SAAProblem(model, samples, beta=beta, tv_rho=tv_rho,
                                      MultipleShooting=False, integrator=integrator,
                                      parallelization=parallelization,
                                      ipopt_options=_IPOPT)


def _numpy_objective(u_seq, nintervals=_NINTERVALS, final_time=1.0, alpha=1.0,
                     steps_per_interval=4, init=(1.0, 1.0)):
    # Independent RK4 rollout of the DoubleIntegrator single-shooting SAA
    # objective for a control sequence.  Dynamics xdot=[v, u], running cost
    # L=alpha/2 u^2, terminal cost |x|^2/2.  The ensemble is degenerate (dynamics
    # ignore the parameter), so the SAA mean equals this single rollout.
    dt = final_time / nintervals / steps_per_interval
    h, v = init
    cost = 0.0
    for u in np.asarray(u_seq, dtype=float).ravel():
        for _ in range(steps_per_interval):
            # RK4 stages for [h, v] with xdot=[v, u] (v-derivative is constant u)
            h = h + dt / 6 * ((v) + 2 * (v + dt / 2 * u)
                              + 2 * (v + dt / 2 * u) + (v + dt * u))
            v = v + dt * u
            cost += dt * (alpha / 2.0) * u * u
    return cost + 0.5 * (h * h + v * v)


def test_objective_matches_numpy_reference():
    # The mapped objective at nontrivial controls must match an independent numpy
    # RK4 rollout -- validates build_interval and the map assembly.
    saa = _build()
    ncontrols = saa.control_problem.ncontrols
    rng = np.random.default_rng(0)
    for _ in range(5):
        u = rng.uniform(-1.0, 1.0, _NINTERVALS * ncontrols)
        assert abs(_scalar(saa.obj(u)) - _numpy_objective(u)) < 1e-9


def test_objective_at_initial_guess_is_known():
    # Initial controls are all zero (midpoint of the unbounded box), so the
    # trajectory is h(t)=1+t, v=1: terminal cost |x(1)|^2/2 = (2^2+1)/2 = 2.5.
    saa = _build()
    assert abs(_scalar(saa.obj(saa.initial_decisions)) - 2.5) < 1e-9


def test_gradient_matches_finite_difference():
    saa = _build()
    ncontrols = saa.control_problem.ncontrols
    rng = np.random.default_rng(1)
    w = rng.uniform(-1.0, 1.0, _NINTERVALS * ncontrols)
    g = saa.derivative(w)
    eps = 1e-6
    fd = np.empty_like(w)
    for i in range(len(w)):
        wp = w.copy()
        wm = w.copy()
        wp[i] += eps
        wm[i] -= eps
        fd[i] = (_scalar(saa.obj(wp)) - _scalar(saa.obj(wm))) / (2 * eps)
    assert np.linalg.norm(g - fd) < 1e-6


def test_thread_backend_matches_serial_backend():
    # The production "thread" backend must give the same optimum as the "serial"
    # map backend -- covers gradient accumulation across threads.
    ws, fs = _build(parallelization="serial").solve()
    wt, ft = _build(parallelization="thread").solve()
    assert abs(_scalar(fs) - _scalar(ft)) < 1e-6
    assert np.max(np.abs(np.asarray(ws) - np.asarray(wt))) < 1e-6


def test_decision_layout_is_controls_only():
    # Risk-neutral single shooting: the decision vector is exactly the controls.
    saa = _build()
    assert len(saa.initial_decisions) == _NINTERVALS * saa.control_problem.ncontrols
    assert saa.constraints.numel() == 0


def test_cvar_adds_tail_and_solves():
    # CVaR adds VaR t plus one slack r_i per sample and one epigraph constraint
    # per sample.
    saa = _build(beta=0.9)
    nsamples = len(_SAMPLES)
    assert len(saa.initial_decisions) == _NINTERVALS + 1 + nsamples
    assert saa.constraints.numel() == nsamples
    w, _ = saa.solve()
    assert np.all(np.isfinite(w))


def test_tv_adds_slacks_and_solves():
    # TV adds a (v, s) slack pair per control per consecutive-interval gap and one
    # equality constraint per gap.
    saa = _build(tv_rho=1e-2)
    ncontrols = saa.control_problem.ncontrols
    gaps = _NINTERVALS - 1
    assert len(saa.initial_decisions) == _NINTERVALS * ncontrols + 2 * ncontrols * gaps
    assert saa.constraints.numel() == ncontrols * gaps
    w, _ = saa.solve()
    assert np.all(np.isfinite(w))


def test_scalar_1d_samples():
    # A 1-D scalar-parameter list ([0, 0] -> 2 samples) must be accepted.
    saa = _build(samples=[0, 0])
    w, _ = saa.solve()
    assert np.all(np.isfinite(w))
    assert abs(_scalar(saa.obj(saa.initial_decisions)) - 2.5) < 1e-9


def test_euler_scheme_solves():
    # The Euler scheme is supported directly by the map (no serial fallback).
    w, _ = _build(integrator="euler").solve()
    assert np.all(np.isfinite(w))


def test_terminal_constraint_is_enforced():
    # Pin one ensemble member's terminal first-state to a target and check the
    # solved trajectory hits it (the point of terminal_constraints).
    target = 0.7
    model = DoubleIntegrator()
    model.nintervals = _NINTERVALS
    con = ensemblecontrol.SAAProblem(
        model, _SAMPLES, MultipleShooting=False, parallelization="serial",
        ipopt_options={"print_level": 0, "sb": "yes", "tol": 1e-9},
        terminal_constraints=[(0, lambda x: x[0], target, target)])

    # one extra scalar constraint vs the unconstrained problem
    assert con.constraints.numel() == _build().constraints.numel() + 1

    w_opt, _ = con.solve()
    plotter = ensemblecontrol.SolutionPlotter(con, w_opt)
    assert abs(plotter.state_mean[0][-1] - target) < 1e-5
