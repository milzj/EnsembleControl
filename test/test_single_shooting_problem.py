
import numpy as np
import ensemblecontrol
from casadi import *


def test_single_shooting_problem():
    # https://github.com/casadi/casadi/blob/main/docs/examples/python/direct_single_shooting.py

    T = 1. # Time horizon
    N = 10 # number of control intervals
    mesh_width = T/N
    initial_state = [1.0]

    lbu = -1.
    ubu = 1.

    # Declare model variables
    x = MX.sym('x')
    u = MX.sym('u')

    # Model equations
    xdot = x + u

    # Objective term
    L = u**2

    # right-hand side and objective
    f = Function('f', [x, u], [xdot, L])

    X = MX.sym("X")
    U = MX.sym("U")

    V, W = f(X, U)
    X += mesh_width*V
    Q += mesh_width*W

    # Discrete time dynamics function
    dynamics = Function('F', [X,U], [xj], ["x0", "p"], ["xf", "qf"])

    # min x(t_f)
    objective_function = lambda x: x

    objective, constraints, decisions, w0, lbw, ubw, lbg, ubg =
                ensemblecontrol.SingleShootingProblem(objective_function,
                                                        dynamics, initial_state,
                                                        [lbu, ubu], 1, N)

    assert np.linalg.norm(w0) == 0.0
    assert objective(decisions) == 1.0
