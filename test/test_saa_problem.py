from casadi import *
import numpy as np
import ensemblecontrol

from .double_integrator import DoubleIntegrator


def test_control_problem():

    double_integrator = DoubleIntegrator()
    samples = [0]

    saa_problem = ensemblecontrol.SAAProblem(double_integrator, samples)
    w_opt, f_opt = saa_problem.solve()

    assert f_opt == -1.0
