import ensemblecontrol
from casadi import *
import numpy as np

class TwoStageCSTR(ensemblecontrol.ControlProblem):
    # Problem 3 (Two-stage CSTR) from the Appendix (p. 10-11) of
    #   Andres-Martinez, O., Biegler, L.T., Flores-Tlacuahuac, A. (2020).
    #   An indirect approach for singular optimal control problems.
    #   Computers and Chemical Engineering 139, 106923.
    #   https://doi.org/10.1016/j.compchemeng.2020.106923
    # Original source of the model:
    #   Sun, D.Y. (2010). The solution of singular optimal control problems using
    #   the modified line-up competition algorithm with region-relaxing strategy.
    #   ISA Transactions 49(1), 106-113.
    #
    # Uncertain parameters: every non-integer constant appearing in the dynamics
    # (fractional-valued constants plus the Arrhenius frequency factors 1.5e7/1.5e10),
    # deduplicated by value.  Integer coefficients, initial conditions and the final
    # time are kept fixed.

    def __init__(self):

        super().__init__()

        self._alpha = 0.0
        self._nintervals = 50
        self._final_time = 0.32498
        self._ncontrols = 2
        self._nstates = 4

        self._control_bounds = [[-1, -1], [1, 1]]

        self.u = MX.sym("u", 2)
        self.x = MX.sym("h", 4)
        self.L = 0.0
        self._nominal_param = [[11.1558, 8.1558, 0.1592,      # k0, k1, k2  (zdot2)
                                1.5, 0.5,                     # k3, k4      (zdot3)
                                0.75, 4.9385, 3.4385, 0.122,  # k5..k8      (zdot4)
                                1.5e7, 1.5e10,                # k9, k10     (frequency factors)
                                0.521, 0.6932, 0.4748, 1.4280,   # k11..k14 (f1)
                                0.4263, 0.6560, 0.5764, 0.5086]] # k15..k18 (f2)
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

        z1, z2, z3, z4 = x[0], x[1], x[2], x[3]
        u1, u2 = u[0], u[1]

        # Note: the paper prints f1's second exponential with -10, but by symmetry
        # with f2 it is implemented here as -15 (both exponents are integers, so
        # this does not change the uncertain-parameter set).
        f1 = k[9]*(k[11] - z1)*exp(-10/(z2 + k[12])) \
             - k[10]*(k[13] + z1)*exp(-15/(z2 + k[12])) - k[14]
        f2 = k[9]*(k[15] - z2)*exp(-10/(z4 + k[16])) \
             - k[10]*(k[17] + z3)*exp(-15/(z4 + k[16])) - k[18]

        xdot = vertcat(
            -3*z1 + f1,
            -k[0]*z2 + f1 - k[1]*(z2 + k[2])*u1,
            k[3]*(k[4]*z1 - z3) + f2,
            k[5]*z2 - k[6]*z4 + f2 - k[7]*(z4 + k[8])*u2,
        )
        self.xdot = xdot

        return Function('f', [x, u, k], [xdot])

    @property
    def integral_cost_function(self):
        return self.L

    def parameterized_initial_state(self, params):
        # parameterized initial value
        return [0.1962, -0.0372, 0.0946, 0.0]

    def final_cost_function(self, x):
        # Objective function to be evaluated
        # at states at final time
        # Notation F in manuscript
        return dot(x, x)
