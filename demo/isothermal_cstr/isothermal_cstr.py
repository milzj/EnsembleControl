import ensemblecontrol
from casadi import *
import numpy as np

class IsothermalCSTR(ensemblecontrol.ControlProblem):
    # Problem 4 (Isothermal CSTR) from the Appendix (p. 10-11) of
    #   Andres-Martinez, O., Biegler, L.T., Flores-Tlacuahuac, A. (2020).
    #   An indirect approach for singular optimal control problems.
    #   Computers and Chemical Engineering 139, 106923.
    #   https://doi.org/10.1016/j.compchemeng.2020.106923
    # Original source of the model:
    #   Balsa-Canto, E., Banga, J.R., Alonso, A.A., Vassiliadis, V.S. (2001).
    #   Dynamic optimization of chemical and biochemical processes using restricted
    #   second-order information. Computers & Chemical Engineering 25(4), 539-546.
    #
    # Uncertain parameters: every non-integer constant appearing in the dynamics,
    # deduplicated by value.  Integer coefficients, initial conditions and the final
    # time are kept fixed.

    def __init__(self):

        super().__init__()

        self._alpha = 0.0
        self._nintervals = 50
        self._final_time = 0.2
        self._ncontrols = 3
        self._nstates = 8

        self._control_bounds = [[0, 0, 0], [20, 6, 4]]

        self.u = MX.sym("u", 3)
        self.x = MX.sym("h", 8)
        self.L = 0.0
        self._nominal_param = [[17.6, 35.2, 51.3, 102.6,  # k0..k3
                                5.8, 3.7, 4.1, 0.09]]     # k4..k7
        self.params = MX.sym("k", len(self._nominal_param[0]))

    @property
    def control_bounds(self):
        # lower and upper bounds
        return self._control_bounds

    @property
    def nominal_param(self):
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

        z1, z2, z3, z4, z5, z6, z7 = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
        u1, u2, u3 = u[0], u[1], u[2]

        q = u1 + u2 + 6

        xdot = vertcat(
            6 - q*z1 - k[0]*z1*z2 - 23*z1*z6*u3,
            u1 - q*z2 - k[0]*z1*z2 - 146*z2*z3,
            u2 - q*z3 - 73*z2*z3,
            -q*z4 + k[1]*z1*z2 - k[2]*z4*z5,
            -q*z5 + 219*z2*z3 - k[2]*z4*z5,
            -q*z6 + k[3]*z4*z5 - 23*z1*z6*u3,
            -q*z7 + 46*z1*z6*u3,
            k[4]*(q*z1 - 6) - k[5]*u1 - k[6]*u2
            + q*(23*z4 + 11*z5 + 28*z6 + 35*z7) - 5*u3**2 - k[7],
        )
        self.xdot = xdot

        return Function('f', [x, u, k], [xdot])

    @property
    def integral_cost_function(self):
        return self.L

    def parameterized_initial_state(self, params):
        # parameterized initial value
        return [0.1883, 0.2507, 0.0467, 0.0899, 0.1804, 0.1394, 0.1046, 0.0]

    def final_cost_function(self, x):
        # Objective function to be evaluated
        # at states at final time
        # Notation F in manuscript
        return -x[7]
