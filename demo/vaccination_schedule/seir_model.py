import ensemblecontrol
from casadi import *
import numpy as np

class SEIRModel(ensemblecontrol.ControlProblem):
    # Based on section 4 in https://doi.org/10.1090/dimacs/075/03
    def __init__(self):

        super().__init__()

        self._alpha = 2.0
        self._nintervals = 50
        self._final_time = 20.
        self._ncontrols = 1
        self._nstates = 5

        self._control_bounds = [[0], [0.9]]

        self.u = MX.sym("u", 1)
        self.x = MX.sym("h", 6)
        self.L = 0.0
        self._nominal_param = [[0.2, 0.525, 0.001, 0.5, 0.5, 0.1]] # a, b, c, d, e, g
        self.params = MX.sym("k", 6)

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

        a, b, c, d, e, g = k[0], k[1], k[2], k[3], k[4], k[5]
        S, E, I, R, N = x[0], x[1], x[2], x[3], x[4]

        xdot = vertcat(\
                        b*N-d*S-c*S*I-u*S,
                        c*S*I-(e+d)*E,
                        e*E-(g+a+d)*I,
                        g*I-d*R+u*S,
                        (b-d)*N-a*I,
                        .1*I+(alpha/2)*u**2)

        self.xdot = xdot
        return Function('f', [x, u, k], [xdot])

    @property
    def integral_cost_function(self):
        return self.L

    def parameterized_initial_state(self, params):
        # parameterized initial value
        S0 = 1000; E0 = 100; I0 = 50; R0 = 15
        return [S0, E0, I0, R0, S0+E0+I0+R0, 0]

    def final_cost_function(self, x):
        # Objective function to be evaluated
        # at states at final time
        # Notation F in manuscript
        return x[5]

