from casadi import *
import numpy as np

class ControlProblem(object):

    def __init__(self):

        return True

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value

    @property
    def nintervals(self):
        return self._nintervals

    @nintervals.setter
    def nintervals(self, value):
        self._nintervals = value

    @property
    def final_time(self):
        return self._final_time

    @final_time.setter
    def final_time(self, value):
        self._final_time = value

    @property
    def control_bounds(self):
        # lower and upper bounds
        return NotImplementedError()

    @property
    def ncontrols(self):
        lower_bounds = self.control_bounds[0]
        return len(lower_bounds)

    @property
    def nominal_param(self):
        return NotImplementedError()

    @property
    def nparams(self):
        return len(self.nominal_param)

    @property
    def nstates(self):
        return len(self.parameterized_initial_state(self.nparams[0]*[0]))

    @property
    def mesh_width(self):
        return self.final_time/self.nintervals

    @property
    def control(self):
        return NotImplementedError()

    @property
    def state(self):
        return NotImplementedError()

    @property
    def right_hand_side(self, x, u, params):
        return NotImplementedError()

    @property
    def integral_cost_function(self):
        return NotImplementedError()

    def parameterized_initial_state(self, params):
        # parameterized initial value
        return NotImplementedError()

    def final_cost_function(self, x):
        # Objective function to be evaluated
        # at states at final time
        # Notation F in manuscript
        return NotImplementedError()

