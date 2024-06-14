import ensemblecontrol
from casadi import *
import numpy as np

class HarmonicOscillator(ensemblecontrol.ControlProblem):
    # Based on Problem S_C in https://doi.org/10.1137/140983161

    def __init__(self):

        super().__init__()

        self._alpha = 1e-3
        self._nintervals = 50
        self._final_time = 1.
        self._ncontrols = 2
        self._nstates = 2

        self._control_bounds = [[-3., -3], [3., 3.]]

        self.u = MX.sym("u", 2)
        self.x = MX.sym("h", 2)
        self.params = MX.sym("k", 1)
        self.L = (self.alpha/2)*dot(self.u, self.u)
        self._nominal_param = [0]
        self._param_initial_state = [np.pi]

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

        xdot = vertcat(-k[0]*x[1]+u[0], k[0]*x[0]+u[1])
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
        return dot(x,x)/2

