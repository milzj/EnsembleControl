from casadi import *
import numpy as np

class HarmonicOscillator(object):

    def __init__(self):

        self.final_time = 1. # Time horizon
        self.lbu = [-3.,-3.]
        self.ubu = [3.,3.]

        self._nintervals = 50
        self._alpha = 1e-3

        self.nominal_param = [np.pi]
        self.nparams = len(self.nominal_param)
        self.ncontrols = len(self.lbu)

        self.nstates = len(self.parameterized_initial_value(self.nparams*[0]))

    @property
    def nintervals(self):
        return self._nintervals

    @nintervals.setter
    def nintervals(self, value):
        self._nintervals = value

    @property
    def mesh_width(self):
        return self.final_time/self.nintervals

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value

    @property
    def parameterized_rhs(self):

        # Declare model variables
        x = MX.sym('x', self.nstates)
        u = MX.sym('u', self.ncontrols)
        k = MX.sym('k', self.nparams)

        # Model equations
        #xdot = vertcat(-k[0]*x[1]+1e-1*(k[1]-np.pi)*x[0]*u[0], k[0]*x[0]+1e-1*(k[2]-np.pi)*x[1]*u[1])
        xdot = vertcat(-k[0]*x[1], k[0]*x[0])

        # Objective term
        L = u[0]**2+u[1]**2

        # parameterized right-hand side
        parametric_rhs = Function('fp', [x, u, k], [xdot])

        return parametric_rhs, L, u

    def parameterized_initial_value(self, params):
        # parameterized initial value
        return [1.0,0.0]

    def objective_function(self, x):
        # Objective function to be evaluated
        # at states at final time
        # Notation F in manuscript
        return x[0]**2 + x[1]**2

