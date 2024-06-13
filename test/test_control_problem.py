from casadi import *
import numpy as np
import ensemblecontrol

class DoubleIntegrator(ensemblecontrol.ControlProblem):

    def __init__(self):

        super().__init__()

        self._alpha = .5
        self._nintervals = 100
        self._final_time = 1.
        self._ncontrols = 1
        self._nstates = 2

        self._control_bounds = [[-inf], [inf]]

        self.u = MX.sym("u")
        self.h = MX.sym("h")
        self.v = MX.sym("v")
        self.x = vertcat(self.h,self.v)
        self.xdot = vertcat(self.v,self.u)
        self.L = (self.alpha/2)*dot(self.u, self.u)
        self._nominal_param = [0]
        self._param_initial_state = [1.0]

    def control_bounds(self):
        # lower and upper bounds
        return self._control_bounds

    @property
    def ncontrols(self):
        return len(self.control_bounds[0])

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
        return self.xdot

    @property
    def control_regularizer(self):
        return self.L

    def parameterized_initial_state(self, params):
        # parameterized initial value
        return [1.0]

    def objective_function(self, x):
        # Objective function to be evaluated
        # at states at final time
        # Notation F in manuscript
        return dot(x,x)/2



def test_control_problem():


    double_integrator = DoubleIntegrator()

    assert double_integrator.nstates == 2
    assert double_integrator.ncontrols == 1
    assert double_integrator.alpha == .5
    assert double_integrator.final_time == 1.
    assert double_integrator.nparams == 1
    assert double_integrator.mesh_width == 0.01

    x = double_integrator.x
    u = double_integrator.u
    xdot = double_integrator.xdot
    L = self.L
    f = Function('f', [x, u], [xdot, L])

    assert f([1,1], [1.0])[1] == .5
    assert f([1,1], [1.0])[0][0] == 1
    assert f([1,1], [1.0])[0][1] == 1
