import numpy as np

import ensemblecontrol
from ensemblecontrol import ScipyBoxSolver

from .double_integrator import DoubleIntegrator


def _taylor_first_order_rates(fun, jac, x, direction, ts):
    """Observed convergence order of the first-order Taylor remainder.

    For a correct gradient g = grad fun(x), the remainder
        r(t) = |fun(x + t d) - fun(x) - t * <g, d>|
    is O(t^2), so halving t divides r by ~4 and the observed order -> 2. A wrong
    (or wrongly scaled) gradient leaves an O(t) term and the order drops to 1.
    """
    x = np.asarray(x, dtype=float)
    direction = np.asarray(direction, dtype=float)
    f0 = fun(x)
    g0 = float(np.dot(np.asarray(jac(x), dtype=float).ravel(), direction))
    rem = np.array([abs(fun(x + t * direction) - f0 - t * g0) for t in ts])
    ts = np.asarray(ts, dtype=float)
    return rem, np.log(rem[:-1] / rem[1:]) / np.log(ts[:-1] / ts[1:])


def test_objective_gradient_taylor():
    # The Euclidean objective gradient d(objective)/d(decisions) that both drivers
    # consume must be a genuine derivative of the SAA objective.
    di = DoubleIntegrator()
    saa = ensemblecontrol.SAAProblem(di, [0], MultipleShooting=False)

    n = saa.decisions.numel()
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n)
    d = rng.standard_normal(n)

    fun = lambda w: float(saa(w))
    jac = lambda w: np.asarray(saa.derivative(w), dtype=float)

    ts = [1e-1 * 0.5 ** k for k in range(7)]
    rem, rates = _taylor_first_order_rates(fun, jac, x, d, ts)
    # first-order remainder converges at order ~2 -> gradient is correct
    assert np.all(rates > 1.9)


def test_scipy_scaled_gradient_taylor():
    # The mass change of variables v = D w wraps the AD gradient with the diagonal
    # chain-rule factor (jac_v = jac / D). Verify _jac really is the gradient of
    # _fun in the scaled v-space, for the preconditioned and the Euclidean driver.
    di = DoubleIntegrator()
    saa = ensemblecontrol.SAAProblem(di, [0], MultipleShooting=False)

    n = saa.decisions.numel()
    rng = np.random.default_rng(1)
    w = rng.standard_normal(n)
    d = rng.standard_normal(n)

    for precondition in (True, False):
        solver = ScipyBoxSolver(saa, precondition=precondition, verbose=False)
        v = solver.d * w                      # a point in the scaled space
        ts = [1e-1 * 0.5 ** k for k in range(7)]
        rem, rates = _taylor_first_order_rates(solver._fun, solver._jac, v, d, ts)
        assert np.all(rates > 1.9)
