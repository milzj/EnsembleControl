from casadi import *
import numpy as np

from .mapped_common import _samples_2d, build_interval, SCHEMES
from .mapped_single_shooting import MappedSingleShootingProblem
from .mapped_multiple_shooting import MappedMultipleShootingProblem
from .idx_state_control import idx_state_control


class SAAProblem(object):

    def __init__(self, control_problem, samples, beta=0.0, MultipleShooting=True,
                 tv_rho=0.0, ipopt_options=None, tol=None, integrator="rk4",
                 lbfgs=True, parallelization="thread", n_threads=None,
                 expand=True, steps_per_interval=4, terminal_constraints=None,
                 precondition=False):

        self.control_problem = control_problem
        self.samples = samples
        self.nsamples = len(samples)
        self.beta = beta
        self.tv_rho = tv_rho
        self.ipopt_options = ipopt_options

        # Deterministic terminal-state constraints on individual ensemble members.
        # Each entry is
        #     (sample, func, lb, ub)
        # enforcing  lb <= func(x^(sample)(t_f)) <= ub, where func maps a terminal
        # state vector to an MX expression. Lets you constrain one (or a few)
        # ensemble members' endpoints, e.g. ethanol's x4(t_f) <= 200.
        self.terminal_constraints = terminal_constraints
        # Ipopt convergence tolerance. None leaves Ipopt's own default (1e-8);
        # set a float to override (also settable via ipopt_options['tol']).
        self.tol = tol

        # Time-integration scheme for the per-sample rollout: "rk4" or "euler".
        if integrator not in SCHEMES:
            raise ValueError("Unknown integrator %r; choose one of %s."
                             % (integrator, list(SCHEMES)))
        self.integrator = integrator
        self.lbfgs = lbfgs

        # Optimize in the mass-scaled variables v = D w (D = sqrt(M)) so Ipopt
        # searches in the discretized-L2 inner product instead of the Euclidean
        # one. Default off (plain Euclidean solve); set True for the L2 metric.
        # On a uniform grid M = h*I is a uniform scalar, so this only rescales
        # conditioning/tolerance (same optimum); the reported criticality is L2-
        # correct regardless of this flag.
        self.precondition = precondition

        # RK4/Euler sub-steps per control interval (finer integration for stiff
        # dynamics), applied consistently by the transcription and the simulator.
        self.steps_per_interval = steps_per_interval

        # The ensemble SAA is always transcribed with the threaded, per-sample
        # CasADi map (see mapped_single_shooting / mapped_multiple_shooting)
        # instead of one stacked symbolic graph. These knobs tune the map.
        self.parallelization = parallelization
        self.n_threads = n_threads
        self.expand = expand

        self.MultipleShooting = MultipleShooting

        self.optimization_problem()

        self.obj = Function("objective", [self.decisions], [self.objective])
        derivative = jacobian(self.objective, self.decisions)
        self.deriv = Function("objective", [self.decisions], [derivative])


    def ensemble_state_trajectory(self, controls):
        """Forward-simulate the ensemble state trajectory for a control sequence.

        Rolls the per-sample interval integrator (the same one the transcription
        uses) over the horizon, sharing the given control across samples. Returns
        an (nstates*nsamples, nintervals+1) array laid out sample-contiguous
        (rows [i*nstates : (i+1)*nstates] are sample i), matching the ensemble
        initial-state ordering. Used to reconstruct single-shooting trajectories
        for plotting, where the states are not decision variables.
        """
        cp = self.control_problem
        samples = _samples_2d(self.samples)
        nsamples = self.nsamples
        nstates = cp.nstates
        ncontrols = cp.ncontrols
        nintervals = cp.nintervals
        nparams = samples.shape[1]

        interval = build_interval(cp, nparams, self.steps_per_interval,
                                  self.integrator)
        if self.expand:
            interval = interval.expand()
        # roll all samples in lock-step, control shared across the ensemble
        sim = interval.map("ensemble_sim", "serial", nsamples,
                           ["u"], [], {"max_num_threads": 1})

        u = np.asarray(controls, dtype=float).reshape(nintervals, ncontrols)
        P = samples.T
        xk = np.array([cp.parameterized_initial_state(samples[i])
                       for i in range(nsamples)], dtype=float).T   # nstates x nsamples

        traj = np.empty((nstates * nsamples, nintervals + 1))
        traj[:, 0] = xk.T.flatten()
        for k in range(nintervals):
            xk = np.array(sim(x0=DM(xk), u=DM(u[k]), k=DM(P))["xf"],
                          dtype=float).reshape(nstates, nsamples)
            traj[:, k + 1] = xk.T.flatten()
        return traj

    def control_matrix(self, w_opt):
        """Extract the control trajectory from a decision vector w_opt.

        Returns an (nintervals, ncontrols) array, for both single and multiple
        shooting. In single shooting the controls are the front block of w_opt;
        in multiple shooting they are interleaved with the lifted states and
        pulled out with idx_state_control (the same extraction SolutionPlotter
        uses). This is the single, matplotlib-free source of truth for reading
        controls out of a decision vector, used by the inference routines.
        """
        cp = self.control_problem
        nstates = cp.nstates
        ncontrols = cp.ncontrols
        nintervals = cp.nintervals
        w = np.asarray(w_opt, dtype=float).ravel()

        if not self.MultipleShooting:
            return w[:nintervals * ncontrols].reshape(nintervals, ncontrols)

        _, idx_control = idx_state_control(nstates, ncontrols, self.nsamples,
                                           nintervals)
        controls = np.empty((ncontrols, nintervals))
        for c in range(ncontrols):
            controls[c] = w[idx_control[c::ncontrols]].flatten()
        return controls.T

    def subproblem(self, indices):
        """A new SAAProblem restricted to samples[indices], reusing this
        problem's configuration.

        Used by the subsampling confidence interval to re-solve the SAA on a
        subset of the scenarios. terminal_constraints reference absolute sample
        indices and cannot be remapped generically, so a problem carrying them
        is rejected here; supply a custom resolver to the subsampling routine
        instead.
        """
        if self.terminal_constraints is not None:
            raise ValueError(
                "subproblem() cannot restrict a problem with terminal_constraints "
                "(they reference absolute sample indices); pass a custom resolve= "
                "to the subsampling routine instead.")
        samples = _samples_2d(self.samples)[np.asarray(indices, dtype=int)]
        return SAAProblem(self.control_problem, samples,
                          beta=self.beta, MultipleShooting=self.MultipleShooting,
                          tv_rho=self.tv_rho, ipopt_options=self.ipopt_options,
                          tol=self.tol, integrator=self.integrator,
                          lbfgs=self.lbfgs, parallelization=self.parallelization,
                          n_threads=self.n_threads, expand=self.expand,
                          steps_per_interval=self.steps_per_interval,
                          terminal_constraints=None,
                          precondition=self.precondition)

    def initial_from_controls(self, controls):
        """A warm-start decision vector for this problem carrying only the given
        controls.

        Starts from this problem's own default initial_decisions (nominal state
        rollout, default CVaR/TV slacks) and overwrites only the control slots
        with `controls` (nintervals, ncontrols). The states are deliberately NOT
        transferred: they are per-sample decision variables whose block
        dimension is nstates*nsamples, so values from a differently sized problem
        are incompatible. Controls are shared across the ensemble and independent
        of the sample count, so they transfer cleanly. Used to warm-start each
        subsampling solve from the full-sample control.
        """
        cp = self.control_problem
        nstates = cp.nstates
        ncontrols = cp.ncontrols
        nintervals = cp.nintervals
        controls = np.asarray(controls, dtype=float).reshape(nintervals, ncontrols)

        w0 = np.asarray(self.initial_decisions, dtype=float).ravel().copy()
        if not self.MultipleShooting:
            w0[:nintervals * ncontrols] = controls.reshape(-1)
            return w0

        _, idx_control = idx_state_control(nstates, ncontrols, self.nsamples,
                                           nintervals)
        for c in range(ncontrols):
            w0[idx_control[c::ncontrols].flatten()] = controls[:, c]
        return w0

    def optimization_problem(self):

        control_bounds = self.control_problem.control_bounds
        nintervals = self.control_problem.nintervals
        nsamples = self.nsamples

        beta = self.beta
        tv_rho = self.tv_rho

        # The ensemble SAA -- single and multiple shooting alike -- is transcribed
        # with the threaded, per-sample CasADi map.
        Problem = (MappedMultipleShootingProblem if self.MultipleShooting
                   else MappedSingleShootingProblem)

        objective, constraints, decisions, initial_decisions, bound_constraints, \
                metric_weights = \
                                Problem(self.control_problem,
                                        self.samples,
                                        control_bounds,
                                        nsamples,
                                        nintervals,
                                        beta=beta,
                                        tv_rho=tv_rho,
                                        parallelization=self.parallelization,
                                        n_threads=self.n_threads,
                                        expand=self.expand,
                                        steps_per_interval=self.steps_per_interval,
                                        scheme=self.integrator,
                                        terminal_constraints=self.terminal_constraints)

        self.objective = objective
        self.decisions = decisions
        self.constraints = constraints
        self.initial_decisions = initial_decisions
        self.bound_constraints = bound_constraints

        # Diagonal of the discretized-L2 mass matrix M over the decision vector
        # (mesh width h on control DOFs, 1 on lifted states / CVaR-TV slacks) and
        # its symmetric factor D = sqrt(M). The change of variables v = D w makes
        # the optimizer's Euclidean geometry on v the L2 geometry on w. See solve()
        # and ScipyBoxSolver.
        self.metric_weights = np.asarray(metric_weights, dtype=float)
        self.sqrt_mass = np.sqrt(self.metric_weights)


    def __call__(self, decisions):

        return self.obj(decisions)

    def derivative(self, decisions):

        deriv = np.array(self.deriv(decisions)).T
        return deriv[:,0]

    def solve(self, precondition=None):

        if precondition is None:
            precondition = self.precondition

        x0 = np.asarray(self.initial_decisions, dtype=float)
        lbx, ubx, lbg, ubg = self.bound_constraints
        lbx = np.asarray(lbx, dtype=float)
        ubx = np.asarray(ubx, dtype=float)

        if precondition:
            # Change of variables v = D w, D = diag(sqrt(M)): optimize in v so
            # Ipopt's Euclidean geometry is the discretized-L2 geometry on w. The
            # objective/constraints are composed with w = D^{-1} v, so CasADi AD
            # supplies the scaled derivatives automatically (no manual gradients).
            # D > 0 diagonal => the control box maps to a box (bounds scaled by D).
            d = self.sqrt_mass
            v = MX.sym('v', self.decisions.numel())
            w_phys = v / DM(d)
            f_expr = self.obj(w_phys)
            if self.constraints.numel() > 0:
                con = Function('con_scaled', [self.decisions], [self.constraints])
                g_expr = con(w_phys)
            else:
                g_expr = self.constraints
            x_sym, x0_solve = v, d * x0
            lbx_solve, ubx_solve = d * lbx, d * ubx
        else:
            f_expr, g_expr, x_sym = self.objective, self.constraints, self.decisions
            x0_solve, lbx_solve, ubx_solve = x0, lbx, ubx

        optimization_problem = {'f': f_expr, 'x': x_sym, 'g': g_expr}

        # Ipopt options. Enable the limited-memory (L-BFGS) Hessian
        # approximation when requested instead of exact second derivatives.
        # Ipopt's own nlp_scaling_method composes on top of the mass metric; set
        # it to 'none' via ipopt_options for the pure L2 metric.
        ipopt_options = dict(self.ipopt_options) if self.ipopt_options is not None else {}
        if self.tol is not None:
            ipopt_options.setdefault('tol', self.tol)
        if self.lbfgs:
            ipopt_options.setdefault('hessian_approximation', 'limited-memory')

        # Solve the NLP
        solver = nlpsol('solver', 'ipopt', optimization_problem, {'ipopt': ipopt_options});
        sol = solver(x0=x0_solve, lbx=lbx_solve, ubx=ubx_solve, lbg=lbg, ubg=ubg)
        v_opt = sol['x'].full().flatten()
        # map back to physical control units
        w_opt = v_opt / self.sqrt_mass if precondition else v_opt

        return w_opt, sol['f'].full().flatten()

