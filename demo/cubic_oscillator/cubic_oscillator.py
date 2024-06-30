import ensemblecontrol
from casadi import *
import numpy as np

class CubicOscillator(ensemblecontrol.ControlProblem):

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
        self.params = MX.sym("k", 4)
        self.L = (self.alpha/2)*dot(self.u, self.u)
        self._nominal_param = [4*[np.pi]]

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
    def nparams(self):
        return 4

    @property
    def right_hand_side(self):

        x = self.x
        u = self.u
        k = self.params
        xdot = vertcat(-k[0]*x[1]+((k[1]-np.pi)/np.pi-x[0]**2-x[1]**2)*x[0]+u[0], k[0]*x[0]+((k[2]-np.pi)/np.pi-x[0]**2-x[1]**2)*x[1]+(k[3]-np.pi)/np.pi+u[1])
        self.xdot = xdot

        return Function('f', [x, u, k], [xdot])

    @property
    def integral_cost_function(self):
        return self.L

    def parameterized_initial_state(self, params):
        # parameterized initial values
        return [1, 0]

    def final_cost_function(self, x):
        # Objective function to be evaluated
        # at states at final time
        # Notation F in manuscript
        return dot(x,x)/2

