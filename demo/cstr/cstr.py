import ensemblecontrol
from casadi import *
import numpy as np

class CSTR(ensemblecontrol.ControlProblem):
    # Based on section 1.8.2 in https://doi.org/10.1201/9780429123641
    # and https://doi.org/10.1002/cjce.5450500231
    def __init__(self):

        super().__init__()

        self._alpha = 1e-6
        self._nintervals = 50
        self._final_time = 0.78
        self._ncontrols = 1
        self._nstates = 3

        #TODO: Increase upper bound to 2
        self._control_bounds = [[0], [5]]

        self.u = MX.sym("u", 1)
        self.x = MX.sym("h", 3)
        self.L = 0.0
        self._nominal_param = [[0.25, 0.5, 25, 2, 0.25, 0.5, 0.5, 25, 2]]
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
        alpha = self.alpha

        x1, x2 = x[0], x[1]

        xdot = vertcat(\
                        -(x1+k[0])+(x2+k[1])*exp(k[2]*x1/(x1+k[3])) - (1+u)*(x1+k[4]),
                        k[5]-x2-(x2+k[6])*exp(k[7]*x1/(x1+k[8])),
                        x1**2+x2**2+alpha/2*u**2
                        )
        self.xdot = xdot

        return Function('f', [x, u, k], [xdot])

    @property
    def integral_cost_function(self):
        return self.L

    def parameterized_initial_state(self, params):
        # parameterized initial value
        return [0.09, 0.09, 0]

    def final_cost_function(self, x):
        # Objective function to be evaluated
        # at states at final time
        # Notation F in manuscript
        return x[2]

