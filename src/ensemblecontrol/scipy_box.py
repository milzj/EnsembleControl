import warnings

from casadi import *
import numpy as np
from scipy.optimize import minimize

from .base import canonical_criticality_measure

__all__ = ["ScipyBoxSolver"]

# scipy.optimize.minimize methods that accept a `bounds` argument and honour the
# control box directly (names are compared lowercased, as scipy does).
_BOUND_METHODS = frozenset({
    'l-bfgs-b', 'tnc', 'slsqp', 'powell', 'trust-constr', 'nelder-mead',
    'cobyqa',
})
# Methods that consume second-order information; for these we hand scipy an exact
# CasADi Hessian of the objective instead of letting it finite-difference one.
_HESSIAN_METHODS = frozenset({
    'newton-cg', 'trust-ncg', 'trust-krylov', 'trust-exact', 'dogleg',
    'trust-constr',
})


class ScipyBoxSolver(object):
    # scipy.optimize.minimize driver for the single-shooting SAA optimal control
    # problem.
    #
    # In single shooting the states are eliminated by forward integration, so the
    # decision vector w is exactly the discretized control trajectory u and the
    # feasible set is the control box
    #
    #     C = { u : lb <= u_k <= ub, k = 0, ..., N-1 }.
    #
    # The SAA objective F(u) and its gradient are supplied by the SAAProblem as
    # the callables saa_problem(u) and saa_problem.derivative(u) (CasADi builds
    # both by algorithmic differentiation), so the box-constrained problem
    #
    #     min_u  F(u)   s.t.   lb <= u_k <= ub
    #
    # is handed straight to scipy without an Ipopt/NLP layer.
    #
    # method selects the scipy algorithm (default 'L-BFGS-B'):
    #   'L-BFGS-B' -- limited-memory BFGS with box bounds. The natural choice
    #       here: it enforces the control box exactly and only needs F and its
    #       gradient. Recommended default.
    #   'Newton-CG' -- truncated-Newton with an exact CasADi Hessian (built
    #       lazily, since Newton-CG uses second-order information). Newton-CG is
    #       an *unconstrained* method: scipy cannot pass it the box, so the bounds
    #       are NOT enforced (a warning is issued). Suitable when the minimizer is
    #       interior; for bang-bang/singular controls that sit on the box faces,
    #       prefer 'L-BFGS-B'.
    # Any other scipy.optimize.minimize method name is accepted and passed
    # through; bounds are supplied only for the methods that support them.
    #
    # max_iter caps the iterations and tol is the gradient tolerance (the
    # projected-gradient max-norm `gtol` for L-BFGS-B/TNC, scipy's top-level
    # `tol` otherwise). tol=None leaves scipy's own default gradient tolerance
    # (gtol=1e-5 for L-BFGS-B); pass a float to override. For L-BFGS-B the
    # relative-function-reduction stop `ftol` is defaulted very small (1e-12):
    # the singular/bang-bang controls here trace a long shallow objective valley,
    # and L-BFGS-B's default ftol would halt on it far from the optimum, so
    # convergence is instead governed by the gradient tolerance. Pass `options`
    # to override or extend the per-method option dict.

    # precondition scales the decision vector by D = sqrt(M) (M the discretized-L2
    # mass, here h*I on the pure-control box) so scipy's Euclidean L-BFGS-B iterates
    # in the discretized-L2 inner product; the default precondition=False is the
    # plain Euclidean solve. The returned control is always in physical units, and
    # the reported criticality is L2-correct regardless of this flag.

    def __init__(self, saa_problem, method='L-BFGS-B', max_iter=1000, tol=None,
                 options=None, verbose=True, precondition=False):

        self.saa_problem = saa_problem
        self.method = method
        self.max_iter = max_iter
        self.tol = tol
        self.options = options
        self.verbose = verbose

        if saa_problem.MultipleShooting:
            raise ValueError(
                "ScipyBoxSolver requires single shooting (MultipleShooting="
                "False) so the decision vector is the control box; build the "
                "SAAProblem with MultipleShooting=False.")

        # The box solver needs a box-only feasible set: the CVaR (beta) and TV
        # (tv_rho) reformulations append equality constraints and unbounded
        # slacks that scipy's bound argument cannot express.
        if saa_problem.constraints.numel() > 0:
            raise ValueError(
                "ScipyBoxSolver needs an unconstrained (box-only) problem; the "
                "SAAProblem carries {} side constraint(s). Build it with beta=0 "
                "and tv_rho=0.".format(saa_problem.constraints.numel()))

        lbw, ubw, _, _ = saa_problem.bound_constraints
        self.lb = np.asarray(lbw, dtype=float)
        self.ub = np.asarray(ubw, dtype=float)
        self.mesh_width = saa_problem.control_problem.mesh_width

        # Change of variables v = d * w, d = sqrt(M) (= sqrt(h) here, since the
        # box-only single-shooting decision vector is pure controls). d = 1
        # recovers the Euclidean solve. Bounds map to the box d*lb <= v <= d*ub.
        self.precondition = precondition
        d = np.asarray(saa_problem.sqrt_mass, dtype=float)
        self.d = d if precondition else np.ones_like(d)
        self.lb_v = self.d * self.lb
        self.ub_v = self.d * self.ub

        self._method_key = method.lower()
        self._use_bounds = self._method_key in _BOUND_METHODS
        if not self._use_bounds and np.any(self.lb > -np.inf) \
                and np.any(self.ub < np.inf):
            warnings.warn(
                "scipy method {!r} does not support bounds; the control box "
                "lb <= u <= ub is NOT enforced. Use 'L-BFGS-B' to respect the "
                "box.".format(method), stacklevel=2)

        self._hess_fun = None
        if self._method_key in _HESSIAN_METHODS:
            H, _ = hessian(saa_problem.objective, saa_problem.decisions)
            self._hess_fun = Function('objective_hessian',
                                      [saa_problem.decisions], [H])

    # -- scipy callables ------------------------------------------------------
    # scipy iterates in the scaled variable v = d * w; these take v, map back to
    # the physical control w = v / d, and apply the (diagonal) chain rule so the
    # objective/gradient/Hessian are consistent in v-space. d = sqrt(M) diagonal
    # => grad_v = grad_w / d, hess_v = hess_w / (d d^T).

    def _fun(self, v):
        return float(self.saa_problem(v / self.d))

    def _jac(self, v):
        g = np.asarray(self.saa_problem.derivative(v / self.d), dtype=float)
        return g / self.d

    def _hess(self, v):
        H = np.asarray(self._hess_fun(v / self.d))
        return H / np.outer(self.d, self.d)

    def _callback(self, *cb_args):
        # scipy passes callback(xk) for most methods and callback(xk, state) for
        # trust-constr; the current iterate is the first argument in both. It is
        # the scaled v; report on the physical control w = v / d.
        w = np.asarray(cb_args[0], dtype=float) / self.d
        f_val = float(self.saa_problem(w))
        crit = canonical_criticality_measure(
            w, np.asarray(self.saa_problem.derivative(w), dtype=float),
            self.lb, self.ub, self.mesh_width)
        self.f_history.append(f_val)
        self.crit_history.append(crit)
        if self.verbose:
            print("scipy {:>9s} iter {:3d}:  F = {: .8e}   crit = {:.3e}"
                  .format(self.method, len(self.f_history), f_val, crit))

    # -- public driver --------------------------------------------------------

    def solve(self, w0=None):

        if w0 is None:
            w = np.asarray(self.saa_problem.initial_decisions, dtype=float)
        else:
            w = np.asarray(w0, dtype=float)
        if self._use_bounds:
            w = np.clip(w, self.lb, self.ub)   # start feasible
        v = self.d * w                          # scaled start handed to scipy

        self.f_history = []
        self.crit_history = []

        options = {'maxiter': self.max_iter}
        # For L-BFGS-B (and TNC) tol is the projected-gradient tolerance gtol;
        # ftol is loosened to a tiny value so the run is not stopped by small
        # per-step function reductions along the shallow singular-arc valley.
        pass_top_level_tol = True
        if self._method_key in ('l-bfgs-b', 'tnc'):
            if self.tol is not None:
                options.setdefault('gtol', self.tol)
            options.setdefault('ftol', 1e-12)
            options.setdefault('maxfun', 100 * self.max_iter)
            pass_top_level_tol = False
        if self.options is not None:
            options.update(self.options)

        kwargs = {
            'method': self.method,
            'jac': self._jac,
            'callback': self._callback,
            'options': options,
        }
        if pass_top_level_tol and self.tol is not None:
            kwargs['tol'] = self.tol
        if self._use_bounds:
            kwargs['bounds'] = list(zip(self.lb_v, self.ub_v))   # scaled box
        if self._hess_fun is not None:
            kwargs['hess'] = self._hess

        result = minimize(self._fun, v, **kwargs)

        self.result = result
        self.w_opt = np.asarray(result.x, dtype=float) / self.d   # physical control
        self.f_opt = float(result.fun)
        self.iterations = int(result.get('nit', len(self.f_history)))
        self.criticality = canonical_criticality_measure(
            self.w_opt,
            np.asarray(self.saa_problem.derivative(self.w_opt), dtype=float),
            self.lb, self.ub, self.mesh_width)

        if self.verbose:
            print("\nscipy {} finished: success={}, {} iterations, "
                  "F = {:.8e}, crit = {:.3e}\n  {}".format(
                      self.method, result.success, self.iterations, self.f_opt,
                      self.criticality, result.message))

        return self.w_opt, self.f_opt
