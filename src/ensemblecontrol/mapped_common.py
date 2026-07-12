"""Building blocks shared by the mapped (threaded) SAA transcriptions.

Both mapped_single_shooting and mapped_multiple_shooting evaluate the ensemble
by mapping a *per-sample* fixed-step integrator over the samples with
Function.map.  This module holds the pieces they have in common: the sample
container normalization and the per-sample single-interval integrator.
"""

import numpy as np
from casadi import *

__all__ = ["_samples_2d", "build_interval", "SCHEMES"]

# Supported per-sample integration schemes (name -> substeps applied per control
# interval): "rk4" (matches RK4Integrator) or explicit "euler".
SCHEMES = ("rk4", "euler")


def _samples_2d(samples):
    # Normalize the sample container to a dense (nsamples, nparams) array.
    # A 1-D list such as [0, 0] means nsamples scalar-parameter samples, i.e.
    # shape (nsamples, 1) -- NOT a single (1, nparams) sample.
    s = np.asarray(samples, dtype=float)
    if s.ndim == 1:
        s = s.reshape(-1, 1)
    return s


def build_interval(control_problem, nparams, steps_per_interval=4, scheme="rk4"):
    """Per-sample integrator for a single control interval.

    Applies ``steps_per_interval`` sub-steps of ``scheme`` (RK4 or explicit
    Euler) to one sample's dynamics.  The uncertain parameter enters as the live
    symbolic input ``k`` (length ``nparams``), so the same function can be mapped
    over the ensemble with each sample's parameter fed as data.

    Returns Function interval(x0[nstates], u[ncontrols], k[nparams]) -> (xf, qf),
    where qf is that sample's integral cost over the interval.
    """
    if scheme not in SCHEMES:
        raise ValueError("Unknown integration scheme %r; choose one of %s."
                         % (scheme, list(SCHEMES)))

    cp = control_problem
    rhs = cp.right_hand_side                        # Function f(x, u, k) -> xdot
    Lfun = Function("L", [cp.state, cp.control, cp.params],
                    [cp.integral_cost_function])
    nstates = cp.nstates
    ncontrols = cp.ncontrols

    DT = cp.final_time / cp.nintervals / steps_per_interval

    X0i = MX.sym("X0", nstates)
    Ui = MX.sym("U", ncontrols)
    Ki = MX.sym("K", nparams)
    X = X0i
    Q = 0.0
    for _ in range(steps_per_interval):
        if scheme == "rk4":
            k1 = rhs(X, Ui, Ki);               q1 = Lfun(X, Ui, Ki)
            k2 = rhs(X + DT / 2 * k1, Ui, Ki); q2 = Lfun(X + DT / 2 * k1, Ui, Ki)
            k3 = rhs(X + DT / 2 * k2, Ui, Ki); q3 = Lfun(X + DT / 2 * k2, Ui, Ki)
            k4 = rhs(X + DT * k3, Ui, Ki);     q4 = Lfun(X + DT * k3, Ui, Ki)
            X = X + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            Q = Q + DT / 6 * (q1 + 2 * q2 + 2 * q3 + q4)
        else:  # "euler"
            k1 = rhs(X, Ui, Ki);               q1 = Lfun(X, Ui, Ki)
            X = X + DT * k1
            Q = Q + DT * q1

    return Function("interval", [X0i, Ui, Ki], [X, Q],
                    ["x0", "u", "k"], ["xf", "qf"])
