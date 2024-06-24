import ensemblecontrol
from casadi import *
import numpy as np

class FedBatchReactor(ensemblecontrol.ControlProblem):
    # Based on https://doi.org/10.1109/9.173155
    # and section 5 in https://doi.org/10.1201/9780429123641
    def __init__(self):

        super().__init__()

        self._alpha = 1.
        self._nintervals = 50
        self._final_time = 20.
        self._ncontrols = 1
        self._nstates = 5

        #TODO: Increase upper bound to 2
        self._control_bounds = [[0], [2]]

        self.u = MX.sym("u", 1)
        self.x = MX.sym("h", 5)
        self.L = self.alpha/2*self.u**2
        self._nominal_param = [[4.75, 0.12, -5, 0.1, 21.87, 0.4, 62.5, -7.3, 20.]]
        self.params = MX.sym("k", len(self._nominal_param[0]))

    @property
    def control_bounds(self):
        # lower and upper bounds
        return self._control_bounds

    @property
    def nominal_param(self):
        # lower and upper bounds
        return self._nominal_param

    @property
    def control(self):
        return self.u

    @property
    def state(self):
        return self.x

    @property
    def right_hand_side(self):

        x = self.x
        u = self.u
        k = self.params
        alpha = self._alpha

        x1, x2, x3, x4, x5 = x[0], x[1], x[2], x[3], x[4]

        g3 = k[4]*x4/(x4+k[5])/(x4+k[6])
        g2 = x4*exp(k[2]*x4)/(k[3]+x4)
        g1 = k[0]*g3/(k[1]+g3)

        xdot = vertcat(\
                        g1*(x2-x1)-u*x1/x5,
                        g2*x3-u/x5*x2,
                        g3*x3-u/x5*x3,
                        k[7]*g3*x3+u/x5*(k[8]-x4),
                        u
                        )
        self.xdot = xdot
        return Function('f', [x, u, k], [xdot])

    @property
    def integral_cost_function(self):
        return self.L

    def parameterized_initial_state(self, params):
        # parameterized initial value
        return [0.0, 0.0, 1.0, 5.0, 1.0]

    def final_cost_function(self, x):
        # Objective function to be evaluated
        # at states at final time
        # Notation F in manuscript
        return -x[4]*x[0]

