"""Correctness of the threaded, per-sample multiple-shooting map transcription.

Multiple shooting solves the same SAA optimal control problem as single shooting
(it just lifts the shooting-node states and adds continuity constraints), so the
primary oracle is agreement between the two mapped transcriptions at the optimum.
Additional tests check the decision layout, that the continuity gaps close, that
the "thread" and "serial" map backends agree, and -- with a parameter-dependent
model -- that each sample's parameter really reaches its own dynamics.
"""
import numpy as np
from casadi import Function, vertcat
import ensemblecontrol

from .double_integrator import DoubleIntegrator

_IPOPT = {"print_level": 0, "sb": "yes", "tol": 1e-9}
_SAMPLES = [[0.0], [0.1], [-0.2]]
_NINTERVALS = 20


def _scalar(x):
    return float(np.asarray(x).ravel()[0])


def _saa(multiple, samples=_SAMPLES, beta=0.0, tv_rho=0.0, integrator="rk4",
         parallelization="serial", model=None, terminal_constraints=None):
    if model is None:
        model = DoubleIntegrator()
    model.nintervals = _NINTERVALS
    return ensemblecontrol.SAAProblem(model, samples, beta=beta, tv_rho=tv_rho,
                                      MultipleShooting=multiple, integrator=integrator,
                                      parallelization=parallelization,
                                      ipopt_options=_IPOPT,
                                      terminal_constraints=terminal_constraints)


def _controls(saa, w_opt):
    # Extract the (ncontrols, nintervals) control trajectory regardless of the
    # shooting discretization via the plotter's extraction logic.
    return ensemblecontrol.SolutionPlotter(saa, w_opt).controls


def _assert_matches_single_shooting(beta=0.0, tv_rho=0.0, samples=_SAMPLES,
                                    model_factory=DoubleIntegrator, tol=1e-4):
    ss = _saa(False, samples, beta, tv_rho, model=model_factory())
    ms = _saa(True, samples, beta, tv_rho, model=model_factory())
    ws, fs = ss.solve()
    wm, fm = ms.solve()
    assert abs(_scalar(fs) - _scalar(fm)) < tol
    assert np.allclose(_controls(ss, ws), _controls(ms, wm), atol=1e-3)


def test_matches_single_shooting_risk_neutral():
    _assert_matches_single_shooting()


def test_matches_single_shooting_cvar():
    _assert_matches_single_shooting(beta=0.9)


def test_matches_single_shooting_tv():
    _assert_matches_single_shooting(tv_rho=1e-2)


def test_matches_single_shooting_one_sample():
    _assert_matches_single_shooting(samples=[[0.0]])


def test_decision_layout():
    # [X0, U0, X1, ..., U_{N-1}, X_N] with each X_k the stacked ensemble state.
    saa = _saa(True)
    nstates = saa.control_problem.nstates
    ncontrols = saa.control_problem.ncontrols
    nsamples = len(_SAMPLES)
    expected = nstates * nsamples * (_NINTERVALS + 1) + ncontrols * _NINTERVALS
    assert len(saa.initial_decisions) == expected
    # continuity: one stacked equality per interval
    assert saa.constraints.numel() == nstates * nsamples * _NINTERVALS
    # idx_state_control indexes stay inside the (risk-neutral) decision vector
    idx_state, idx_control = ensemblecontrol.idx_state_control(
        nstates, ncontrols, nsamples, _NINTERVALS)
    assert idx_state.shape == (nstates * nsamples, _NINTERVALS + 1)
    assert idx_control.shape == (ncontrols, _NINTERVALS)
    assert idx_state.max() < expected and idx_control.max() < expected


def test_continuity_gaps_close_at_optimum():
    saa = _saa(True)
    w_opt, _ = saa.solve()
    g = Function("g", [saa.decisions], [saa.constraints])
    residual = np.asarray(g(w_opt)).ravel()
    assert np.max(np.abs(residual)) < 1e-6


def test_thread_backend_matches_serial_backend():
    ws, fs = _saa(True, parallelization="serial").solve()
    wt, ft = _saa(True, parallelization="thread").solve()
    assert abs(_scalar(fs) - _scalar(ft)) < 1e-6
    assert np.max(np.abs(np.asarray(ws) - np.asarray(wt))) < 1e-5


def test_euler_scheme_solves():
    w, _ = _saa(True, integrator="euler").solve()
    assert np.all(np.isfinite(w))


class _ParamShift(DoubleIntegrator):
    # xdot = [v, u + k]: the scalar parameter shifts the acceleration, so distinct
    # samples give distinct trajectories (a genuinely non-degenerate ensemble).
    @property
    def right_hand_side(self):
        xdot = vertcat(self.v, self.u + self.params[0])
        return Function("f", [self.x, self.u, self.params], [xdot])


def test_parameter_reaches_each_sample():
    # With parameter-dependent dynamics the ensemble must spread, proving each
    # sample's parameter reaches its own mapped dynamics; SS and MS must still
    # agree at the optimum.
    samples = [[0.0], [0.5], [-0.5]]
    _assert_matches_single_shooting(samples=samples, model_factory=_ParamShift)

    ms = _saa(True, samples=samples, model=_ParamShift())
    w_opt, _ = ms.solve()
    plotter = ensemblecontrol.SolutionPlotter(ms, w_opt)
    assert np.max(plotter.state_std) > 1e-3


def test_terminal_constraint_is_enforced():
    # Multiple shooting also supports per-member terminal constraints (read off
    # the final shooting node).
    target = 0.7
    con = _saa(True, terminal_constraints=[(0, lambda x: x[0], target, target)])
    w_opt, _ = con.solve()
    plotter = ensemblecontrol.SolutionPlotter(con, w_opt)
    assert abs(plotter.state_mean[0][-1] - target) < 1e-5
