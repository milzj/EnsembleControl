from casadi import *
import numpy as np

from .rk4_integrator import RK4Integrator
from .ensemble_rhs import EnsembleRHS
from .multiple_shooting_problem import MultipleShootingProblem
from .single_shooting_problem import SingleShootingProblem

class SAAProblem(object):

    def __init__(self, control_problem, samples, MultipleShooting="True"):

        self.control_problem = control_problem
        self.samples = samples
        self.nsamples = len(samples)


        def ensemblerhs(self):


            parameterized_rhs = self.control_problem.parameterized_rhs
            integral_cost_function = self.control_problem.integral_cost_function
            control = self.control_problem.control
            parameterized_initial_state = self.control_problem.parameterized_initial_state
            nstates = self.control_problem.nstates

            ensemble_rhs, ensemble_initial_state = EnsembleRHS(parameterized_rhs,
                                                            integral_cost_function,
                                                            control,
                                                            parameterized_initial_state,
                                                            samples,
                                                            nstates)
            self.ensemble_rhs = ensemble_rhs
            self.ensemble_initial_state = self.ensemble_initial_state

            # dynamics

            self.dynamics = RK4integrator(ensemble_rhs,
                                            final_time,
                                            nintervals,
                                            nensemblestates,
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
                                                control_bounds
                                                nsamples,
                                                nintervals)

            self.objective = objective
            self.decisions = decisions
            self.constraints = constraints
            self.initial_decisions = initial_decisions
            self.bound_constraints = self.bound_constraints


        def solve(self):

            x0 = self.initial_decisions
            lbx, ubx, lbg, ubg = self.bound_constraints

            optimization_problem = {'f': self.objective,
                                    'x': self.decisions,
                                    'g': self.constraints}

            # Solve the NLP
            solver = nlpsol('solver', 'ipopt', optimization_problem);
            sol = solver(x0=initial_decisions, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
            w_opt = sol['x'].full().flatten()

            return w_opt, sol['f']

