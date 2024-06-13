from casadi import *
import numpy as np
import ensemblecontrol

from .double_integrator import DoubleIntegrator


def test_control_problem():

    double_integrator = DoubleIntegrator()

    assert double_integrator.nstates == 2
    assert double_integrator.ncontrols == 1
    assert double_integrator.alpha == 1.
    assert double_integrator.final_time == 1.
    assert double_integrator.nparams == 1
    assert double_integrator.mesh_width == 0.01

    x = double_integrator.x
    u = double_integrator.u
    xdot = double_integrator.xdot
    L = double_integrator.L
    f = Function('f', [x, u], [xdot, L])

    assert f([1,1], [1.0])[1] == .5
    assert f([1,1], [1.0])[0][0] == 1
    assert f([1,1], [1.0])[0][1] == 1
