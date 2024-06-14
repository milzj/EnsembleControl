import ensemblecontrol
from casadi import *
import numpy as np

from .double_integrator import DoubleIntegrator

def test_saa_problem():

    double_integrator = DoubleIntegrator()
    samples = [0]

    saa_problem = ensemblecontrol.SAAProblem(double_integrator, samples)

    f = saa_problem.ensemble_rhs
    ensemble_rhs = f(Y=[1,1], u=[1.])

    assert ensemble_rhs["rhs"][0] == 1
    assert ensemble_rhs["rhs"][1] == 1

    assert 1 == 1
    assert not np.isnan(saa_problem.initial_decisions[2])
    assert not np.isinf(saa_problem.initial_decisions[2])

    assert not np.isnan(saa_problem.initial_decisions).any()

    w_opt, f_opt = saa_problem.solve()

    assert f_opt == saa_problem(w_opt)

    # single shooting optimization problem is unconstrained
    saa_problem = ensemblecontrol.SAAProblem(double_integrator, samples, MultipleShooting=False)
    w_opt, f_opt = saa_problem.solve()

    h = saa_problem.control_problem.mesh_width
    assert ensemblecontrol.base.norm_vec(saa_problem.derivative(w_opt), h) < 1e-8
