import ensemblecontrol
from casadi import *
import numpy as np


class BatchReactor(ensemblecontrol.ControlProblem):
    # First example of Terwiesch & Agarwal (1995), "Robust input policies for
    # batch reactors under parametric uncertainty," Chem. Eng. Commun. 131, 33-52.
    #
    # A temperature-controlled batch reactor running the second-order reaction
    #   2A -> B -> C   (B is the desired product, C an autocatalytic waste).
    # Working in the dimensionless states x1 = [A], x2 = [B] (Eqs. 22-24), the
    # waste [C] is eliminated by the mass balance, so only two ODEs remain. The
    # control is the temperature profile T(t); the uncertain parameter is the
    # collision factor k20 of the second (decomposition) rate constant.
    def __init__(self):

        super().__init__()

        # Physical constants (Sec. 3.1). E1, E2 are activation energies, k10 the
        # (known) collision factor of the main reaction, C0 the initial impurity.
        self._R = 1.987          # cal / (mol K)  -- gas constant
        self._E1 = 3.0e3         # cal / mol      -- activation energy, reaction 1
        self._E2 = 4.0e3         # cal / mol      -- activation energy, reaction 2
        self._k10 = 100.0        # collision factor, reaction 1 (assumed known)
        self._C0 = 5.0e-3        # initial impurity [C]0

        self._alpha = 0.0        # Mayer-only problem: no running cost (L = 0)
        self._nintervals = 50    # stair-function control, Delta t = 0.02 (tf = 1)
        self._final_time = 1.0
        self._ncontrols = 1
        self._nstates = 2

        # Temperature box [340, 420] K (bounds from replicate_first_example.py).
        self._control_bounds = [[340.0], [420.0]]

        self.u = MX.sym("T", 1)          # control: temperature T(t)
        self.x = MX.sym("x", 2)          # states x1 = [A], x2 = [B]
        self.params = MX.sym("k20", 1)   # uncertain collision factor k20
        self.L = (self.alpha / 2) * dot(self.u, self.u)   # 0 (no running cost)
        self._nominal_param = [[1000.0]]                  # nominal k20

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

        x1, x2 = x[0], x[1]
        T = u[0]
        k20 = k[0]

        # Arrhenius rate constants (Eq. 19). k1 is known; k2 carries the
        # uncertain collision factor k20.
        k1 = self._k10 * exp(-self._E1 / (self._R * T))
        k2 = k20 * exp(-self._E2 / (self._R * T))

        # Dimensionless dynamics (Eqs. 22-23); [C] has been eliminated via the
        # mass balance [A] + 2[B] + 2[C] = const with [B]0 = 0.
        dx1 = -2.0 * k1 * x1**2
        dx2 = k1 * x1**2 - 0.5 * k2 * x2 * (1.0 - x1 - 2.0 * x2)

        xdot = vertcat(dx1, dx2)
        self.xdot = xdot
        return Function('f', [x, u, k], [xdot])

    @property
    def integral_cost_function(self):
        return self.L

    def parameterized_initial_state(self, params):
        # x1(0) = 1 - 2[C]0, x2(0) = 0  (Eq. 24)
        return [1.0 - 2.0 * self._C0, 0.0]

    def final_cost_function(self, x):
        # Maximize the final amount of B (Eqs. 21/25): minimize -x2(tf).
        return -x[1]
