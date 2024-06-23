import ensemblecontrol
from casadi import *
import numpy as np

class CatalystMixing(ensemblecontrol.ControlProblem):
    # Based on https://doi.org/10.1007/s10957-014-0641-4

    def __init__(self):

        super().__init__()

        self._alpha = 0.0
        self._nintervals = 100
        self._final_time = 12.
        self._ncontrols = 1
        self._nstates = 2

        self._control_bounds = [[0.], [1.]]

        self.u = MX.sym("u", 1)
        self.x = MX.sym("h", 2)
        self.params = MX.sym("k", 3)
        self.L = self._alpha/2*self.u**2
        self._nominal_param = [[1., 10., 1.]]

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

        xdot = vertcat(-u*(k[0]*x[0]-k[1]*x[1]), u*(k[0]*x[0]-k[1]*x[1])-(1-u)*k[2]*x[1])
        self.xdot = xdot

        return Function('f', [x, u, k], [xdot])

    @property
    def integral_cost_function(self):
        return self.L

    def parameterized_initial_state(self, params):
        # parameterized initial value
        return [1.0, 0.0]

    def final_cost_function(self, x):
        # Objective function to be evaluated
        # at states at final time
        # Notation F in manuscript
        return x[0]+x[1]-1.0

