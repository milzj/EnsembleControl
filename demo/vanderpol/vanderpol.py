import ensemblecontrol
from casadi import *
import numpy as np

class VanDerPol(ensemblecontrol.ControlProblem):
    # 9.5.2.1 SIS Model with Treatment in 
    def __init__(self):

        super().__init__()

        self._alpha = 1e-1
        self._nintervals = 50
        self._final_time = 8
        self._ncontrols = 1
        self._nstates = 2

        self._control_bounds = [[-1], [1]]

        self.u = MX.sym("u", 1)
        self.x = MX.sym("h", 2)
        self.L = (self.alpha/2)*dot(self.u, self.u)
        self._nominal_param = [[1., 1.0, 1.0, 1.0, 1.0]] # beta, gamma, mu
        self.params = MX.sym("k", 5)

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
        return 5
        
    @property
    def right_hand_side(self):

        x = self.x
        u = self.u
        k = self.params

        xdot = vertcat(x[1], x[1]*k[1]*(k[0]-x[0]**2)-k[2]*x[0]+u)
        self.xdot = xdot

        return Function('f', [x, u, k], [xdot])

    @property
    def integral_cost_function(self):
        return self.L

    def parameterized_initial_state(self, params):
        # parameterized initial value
        return [params[3], params[4]]

    def final_cost_function(self, x):
        # Objective function to be evaluated
        # at states at final time
        # Notation F in manuscript
        return dot(x,x)/2

