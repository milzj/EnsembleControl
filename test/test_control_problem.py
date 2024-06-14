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

    f = double_integrator.right_hand_side

    assert f([1,1], [1.0], [0])[0] == 1
    assert f([1,1], [1.0], [0])[1] == 1

    double_integrator.nintervals = 1000
    assert double_integrator.mesh_width == 0.001

    double_integrator.alpha = 0.005
    assert double_integrator.alpha == 0.005

    double_integrator.final_time = 2.
    assert double_integrator.final_time == 2.
