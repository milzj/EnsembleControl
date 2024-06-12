import ensemblecontrol
from casadi import *


def test_rk4integrator():
    # Test based on https://github.com/casadi/casadi/blob/main/docs/examples/python/direct_multiple_shooting.py
    # Evaluates integrator and compares output with
    # that obtained via above script

    T = 10. # Time horizon
    N = 20 # number of control intervals
    ncontrols = 1
    nstates = 2

    # Declare model variables
    x1 = MX.sym('x1')
    x2 = MX.sym('x2')
    x = vertcat(x1, x2)
    u = MX.sym('u')

    # Model equations
    xdot = vertcat((1-x2**2)*x1 - x2 + u, x1)

    # Objective term
    L = x1**2 + x2**2 + u**2

    f = Function('f', [x, u], [xdot, L])

    F = ensemblecontrol.RK4Integrator(f, T, N, nstates, ncontrols)

    # Evaluate at a test point
    Fk = F(x0=[0.2,0.3],p=0.4)

    assert np.isclose(Fk['xf'][0], [0.335539], rtol=1e-4, atol=0.0)
    assert np.isclose(Fk['xf'][1], [0.434784], rtol=1e-4, atol=0.0)
    assert np.isclose(Fk['qf'], [0.183287], rtol=1e-4, atol=0.0)

