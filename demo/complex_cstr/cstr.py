import ensemblecontrol
from casadi import *
import numpy as np

class CSTR(ensemblecontrol.ControlProblem):
    # Based on section 5.2 in https://doi.org/10.1201/9780429123641
    # and  https://doi.org/10.1002/cjce.5450500617
    def __init__(self):

        super().__init__()

        self._alpha = 1e-2
        self._nintervals = 50
        self._final_time = 0.2
        self._ncontrols = 3
        self._nstates = 8

        #TODO: Increase upper bound to 2
        self._control_bounds = [[0, 0, 0], [20, 6, 4]]

        self.u = MX.sym("u", 3)
        self.x = MX.sym("h", 8)
        self.L = 0.0
        self._nominal_param = [[6,
                                  17.6, 73.0, 51.3, 23, 5.8, 3.7, 4.1, 23, 11, 28, 35, 5, 0.099,
                                 .1883, .2507, 0.0467, 0.0899, 0.1804, 0.1394, 0.1046]]
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

        x1, x2, x3, x4, x5, x6, x7 = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
        u1, u2, u3 = u[0], u[1], u[2]

        q1 = k[0]
        k1, k2, k3, k4 = k[1], k[2], k[3], k[4]
        c1, c2, c3, c4, c5, c6, c7, c8, c9 = k[5], k[6], k[7], k[8], k[9], k[10], k[11], k[12], k[13]

        q = q1 + u1 + u2

        xdot = vertcat(\
                        q1-q*x1-k1*x1*x2-k4*x1*x6*u3,
                        u1-q*x2-k1*x1*x2-2*k2*x2*x3,
                        u2-q*x3-k2*x2*x3,
                        -q*x4+2*k1*x1*x2-k3*x4*x5,
                        -q*x5+3*k2*x2*x3-k3*x4*x5,
                        -q*x6+2*k3*x4*x5-k1*x1*x6*u3,
                        -q*x7+2*k4*x1*x6*u3,
                        c1*(q*x1-q1)-c2*u1-c3*u2+q*(c4*x4+c5*x5+c6*x6+c7*x7) - c6*u3**2 - c9 - alpha/2*dot(u,u)
                    )
        self.xdot = xdot

        return Function('f', [x, u, k], [xdot])

    @property
    def integral_cost_function(self):
        return self.L

    def parameterized_initial_state(self, params):
        # parameterized initial value
        #return [.1883, .2507, 0.0467, 0.0899, 0.1804, 0.1394, 0.1046, 0.0]
        return [params[14], params[15], params[16], params[17], params[18], params[19], params[20], 0]

    def final_cost_function(self, x):
        # Objective function to be evaluated
        # at states at final time
        # Notation F in manuscript
        return -x[7]

