
import numpy as np
import ensemblecontrol
from casadi import *


def test_single_shooting_problem():
    # https://github.com/casadi/casadi/blob/main/docs/examples/python/direct_single_shooting.py

    T = 1. # Time horizon
    N = 100 # number of control intervals
    mesh_width = T/N
    initial_state = [1.0]

    lbu = [-1.]
    ubu = [1.]
    alpha = 1e-3

    # Declare model variables
    x = MX.sym('x')
    u = MX.sym('u')

    # Model equations
    xdot = x + u

    # Objective term
    L = (alpha/2)*u**2

    # right-hand side and objective
    f = Function('f', [x, u], [xdot, L])

    X0 = MX.sym("X")
    U = MX.sym("U")
    X = X0
    Q = 0.0

    V, W = f(X, U)
    X += mesh_width*V
    Q += mesh_width*W

    # Discrete time dynamics function
    dynamics = Function('F', [X0,U], [X, Q], ["x0", "p"], ["xf", "qf"])

    # min x(t_f)
    objective_function = lambda x: x

    objective, constraints, decisions, initial_decisions, bound_constraints = \
                ensemblecontrol.SingleShootingProblem(objective_function,
                                                        dynamics, initial_state,
                                                        [lbu, ubu], 1, N)

    obj = Function("objective", [decisions], [objective])

    assert np.linalg.norm(initial_decisions) == 0.0
    # https://www.wolframalpha.com/input?i=use+Euler+method+y%27+%3D+y%2C+y%280%29+%3D+1%2C+from+0+to+1%2C+h+%3D+1%2F100
    assert np.abs(obj(initial_decisions)-2.70481)/2.70481 < 1e-5
    assert len(initial_decisions) == N
