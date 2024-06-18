import ensemblecontrol
from casadi import *
import numpy as np

class DoubleIntegrator(ensemblecontrol.ControlProblem):

    def __init__(self):

        super().__init__()

        self._alpha = 1.0
        self._nintervals = 100
        self._final_time = 1.
        self._ncontrols = 1
        self._nstates = 2

        self._control_bounds = [[-inf], [inf]]

        self.u = MX.sym("u")
        self.h = MX.sym("h")
        self.v = MX.sym("v")
        self.params = MX.sym("p", 1)
        self.x = vertcat(self.h,self.v)
        self.L = (self.alpha/2)*dot(self.u, self.u)
        self._nominal_param = [[0]]
        self._param_initial_state = [1.0]

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
        v = self.v
        u = self.u
        params = self.params
        xdot = vertcat(v, u)
        self.xdot = xdot

        return Function('f', [x, u, params], [xdot])

    @property
    def integral_cost_function(self):
        return self.L

    def parameterized_initial_state(self, params):
        # parameterized initial value
        return [1.0, 1.0]

    def final_cost_function(self, x):
        # Objective function to be evaluated
        # at states at final time
        # Notation F in manuscript
        return dot(x,x)/2

