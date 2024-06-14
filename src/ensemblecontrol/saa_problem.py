from casadi import *
import numpy as np

from .rk4integrator import RK4Integrator
from .ensemblerhs import EnsembleRHS
from .multiple_shooting_problem import MultipleShootingProblem
from .single_shooting_problem import SingleShootingProblem

class SAAProblem(object):

    def __init__(self, control_problem, samples, MultipleShooting="True"):

        self.control_problem = control_problem
        self.samples = samples
        self.nsamples = len(samples)

        self.MultipleShooting = MultipleShooting

        self.ensemblerhs()
        self.optimization_problem()

        self.obj = Function("objective", [self.decisions], [self.objective])
        derivative = jacobian(self.objective, self.decisions)
        self.deriv = Function("objective", [self.decisions], [derivative])


    def ensemblerhs(self):

        right_hand_side = self.control_problem.right_hand_side
        integral_cost_function = self.control_problem.integral_cost_function
        control = self.control_problem.control
        parameterized_initial_state = self.control_problem.parameterized_initial_state
        nstates = self.control_problem.nstates
        final_time = self.control_problem.final_time
        nintervals = self.control_problem.nintervals
        ncontrols = self.control_problem.ncontrols

        samples = self.samples

        ensemble_rhs, ensemble_initial_state = EnsembleRHS(right_hand_side,
                                                        integral_cost_function,
                                                        control,
                                                        parameterized_initial_state,
                                                        samples,
                                                        nstates)
        self.ensemble_rhs = ensemble_rhs
        self.ensemble_initial_state = ensemble_initial_state

        # dynamics
        self.dynamics = RK4Integrator(ensemble_rhs,
                                        final_time,
                                        nintervals,
                                        len(ensemble_initial_state),
                                        ncontrols)

    def optimization_problem(self):

        final_cost_function = self.control_problem.final_cost_function
        control_bounds = self.control_problem.control_bounds
        nintervals = self.control_problem.nintervals

        dynamics = self.dynamics
        ensemble_initial_state = self.ensemble_initial_state
        nsamples = self.nsamples

        if self.MultipleShooting=="True":

            Problem = MultipleShootingProblem

        else:

            Problem = SingleShootingProblem

        objective, constraints, decisions, initial_decisions, bound_constraints = \
                                    Problem(final_cost_function,
                                            dynamics,
                                            ensemble_initial_state,
                                            control_bounds,
                                            nsamples,
                                            nintervals)

        self.objective = objective
        self.decisions = decisions
        self.constraints = constraints
        self.initial_decisions = initial_decisions
        self.bound_constraints = bound_constraints


    def __call__(self, decisions):

        return self.obj(decisions)

    def derivative(self, decisions):

        deriv = np.array(self.deriv(decisions)).T
        return deriv[:,0]

    def solve(self):

        x0 = self.initial_decisions
        lbx, ubx, lbg, ubg = self.bound_constraints

        optimization_problem = {'f': self.objective,
                                'x': self.decisions,
                                'g': self.constraints}

        # Solve the NLP
        solver = nlpsol('solver', 'ipopt', optimization_problem);
        sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        w_opt = sol['x'].full().flatten()

        return w_opt, sol['f']

