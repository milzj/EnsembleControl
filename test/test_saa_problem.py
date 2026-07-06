import ensemblecontrol
from casadi import *
import numpy as np

from .double_integrator import DoubleIntegrator

def test_saa_problem():

    double_integrator = DoubleIntegrator()
    samples = [0]

    saa_problem = ensemblecontrol.SAAProblem(double_integrator, samples)

    assert not np.isnan(saa_problem.initial_decisions[2])
    assert not np.isinf(saa_problem.initial_decisions[2])

    assert not np.isnan(saa_problem.initial_decisions).any()

    w_opt, f_opt = saa_problem.solve()

    assert f_opt == saa_problem(w_opt)

    # single shooting optimization problem is unconstrained. Tighten the Ipopt
    # termination tolerance (below the 1e-5 default) so the solve reaches the
    # first-order optimality this assertion checks.
    saa_problem = ensemblecontrol.SAAProblem(double_integrator, samples,
                                             MultipleShooting=False,
                                             ipopt_options={"tol": 1e-9})
    w_opt, f_opt = saa_problem.solve()

    # First-order optimality in the discretized-L2 metric: the canonical
    # criticality measure uses the Riesz gradient (g/h), so this is metric-correct
    # and mesh-independent, unlike the plain Euclidean gradient norm.
    h = saa_problem.control_problem.mesh_width
    lbw, ubw, _, _ = saa_problem.bound_constraints
    lb = np.asarray(lbw, dtype=float)
    ub = np.asarray(ubw, dtype=float)
    crit = ensemblecontrol.base.canonical_criticality_measure(
        w_opt, saa_problem.derivative(w_opt), lb, ub, h)
    assert crit < 1e-8


def test_saa_problem_cvar():
    # Regression test: with 0 < beta < 1 (CVaR) the initial guess must stay a
    # float-only list. risk_measures.py used to append a casadi DM to w0, which
    # made solve() fail on the solver call. See both shooting discretizations.
    double_integrator = DoubleIntegrator()
    samples = [0, 0]

    for multiple_shooting in [True, False]:
        saa_problem = ensemblecontrol.SAAProblem(double_integrator, samples, beta=0.9,
                                                 MultipleShooting=multiple_shooting)
        # No casadi DM objects leaked into the initial guess.
        initial_decisions = np.array(saa_problem.initial_decisions, dtype=float)
        assert not np.isnan(initial_decisions).any()

        w_opt, f_opt = saa_problem.solve()
        assert np.all(np.isfinite(w_opt))
