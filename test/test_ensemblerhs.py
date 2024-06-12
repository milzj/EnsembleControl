import ensemblecontrol
from casadi import *

def test_ensemblerhs():

        x = MX.sym('x', 2)
        u = MX.sym('u', 2)
        k = MX.sym('k', 1)

        # Model equations
        xdot = vertcat(-k[0]*x[1]+u[0], k[0]*x[0]+u[1])

        # Objective term
        objective = u[0]**2+u[1]**2

        # parameterized right-hand side
        parametric_rhs = Function('fp', [x, u, k], [xdot], ["x", "u", "k"], ["xf"])

        prhs_xuk = parametric_rhs(x=[1.0, 0.0], u=[-1.0, 1.0], k=2.0)
        assert prhs_xuk["xf"][0] == -1.0
        assert prhs_xuk["xf"][1] == 3.0

        parameterized_initial_value = lambda params : [1.0, 0.0]

        samples = [2.0]

        ensemble_rhs, ensemble_initial_value = ensemblecontrol.EnsembleRHS(parametric_rhs, objective, u,
                parameterized_initial_value, samples, 2)

        erhs_xuk = ensemble_rhs(Y=[1.0, 0.0], u=[-1.0, 1.0])
        assert erhs_xuk["rhs"][0] == -1.0
        assert erhs_xuk["rhs"][1] == 3.0

        samples = [2.0, -2.0]

        ensemble_rhs, ensemble_initial_value = ensemblecontrol.EnsembleRHS(parametric_rhs, objective, u,
                parameterized_initial_value, samples, 2)

        erhs_xuk = ensemble_rhs(Y=[1.0, 0.0, 2.0, 0.0], u=[-1.0, 1.0])
        assert erhs_xuk["rhs"][2] == -1.0
        assert erhs_xuk["rhs"][3] == -3.0
