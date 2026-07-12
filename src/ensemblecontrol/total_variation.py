from casadi import *

__all__ = ["tv_regularization"]


def tv_regularization(J, w, w0, lbw, ubw, g, lbg, ubg, controls, tv_rho, ncontrols):
    # Total-variation (TV) regularization of the control, added to the objective as
    #     tv_rho * sum_i sum_k |u_{i,k+1} - u_{i,k}|,
    # which penalizes jumps between successive control values and thereby damps
    # the oscillations that appear along singular arcs.
    #
    # Following Aghaee and Hager (2021), The switch point algorithm, SIAM J.
    # Control Optim. 59(4), 2570-2593 (sect. 5, https://doi.org/10.1137/21M1393315),
    # the nonsmooth absolute value is replaced by its smooth reformulation
    #     |d| = min{ v + s : d = v - s, v >= 0, s >= 0 },
    # introducing one pair of nonnegative slacks (v, s) per control component and
    # per consecutive-interval pair. The resulting problem stays differentiable,
    # so it is handled directly by Ipopt.
    #
    # The slack variables are appended to the tail of the decision vector, past
    # the state/control blocks that idx_state_control indexes, so the extraction
    # of the state and control trajectories is unaffected.
    if tv_rho is None or tv_rho <= 0.0:
        return J, w, lbw, ubw, g, lbg, ubg

    for k in range(len(controls) - 1):
        du = controls[k + 1] - controls[k]

        v = MX.sym('tv_v_' + str(k), ncontrols)
        s = MX.sym('tv_s_' + str(k), ncontrols)
        w += [v, s]
        lbw += 2 * ncontrols * [0.0]
        ubw += 2 * ncontrols * [inf]
        w0 += 2 * ncontrols * [0.0]

        J += tv_rho * (sum1(v) + sum1(s))

        g += [du - (v - s)]
        lbg += ncontrols * [0.0]
        ubg += ncontrols * [0.0]

    return J, w, lbw, ubw, g, lbg, ubg
