"""Singular-control reconstruction for control-affine problems (matplotlib-free).

From a solved single-shooting control, reconstruct the (ensemble) singular
control via the switching function

    phi_j = - sum_s A_s / sum_s B_s,   A_s = mach['A'], B_s = mach['B'],

using per-sample forward states and backward adjoints.  The switching quantities
(a, b, F_x, sigma, phi) are built by CasADi AD from ``model.right_hand_side``, so
this works for any control-affine ``ControlProblem`` (fed-batch reactor,
catalyst-mixing, ethanol-fermentation, ...).  The demos' post-processing plotters
call these; keeping the reconstruction here lets each demo keep its own thin
plotting file without importing another demo's code.

Public API: ``detect_arcs``, ``build_switching_machinery``, ``ensemble_phi_and_control``.
"""

import numpy as np
from casadi import MX, DM, mtimes, jacobian, substitute, Function

__all__ = ["detect_arcs", "build_switching_machinery", "ensemble_phi_and_control"]


# ========================================================================== #
# Arc detection: classify each interval, group into arcs
# ========================================================================== #
def detect_arcs(u, lb, ub, mesh_width, tol_frac=0.02, min_arc_width=None):
    """Label each interval LOW/HIGH/SING by proximity to the bounds and group
    consecutive equal labels into arcs. Narrow arcs are merged away so that
    residual chatter does not create spurious switches.

    Returns an ordered list of dicts: {type, k0, k1, t0, t1}.
    """
    if min_arc_width is None:
        min_arc_width = 5 * mesh_width
    tol = tol_frac * (ub - lb)

    labels = np.full(len(u), "SING", dtype=object)
    labels[np.abs(u - lb) <= tol] = "LOW"
    labels[np.abs(u - ub) <= tol] = "HIGH"

    def group(labels):
        arcs, k0 = [], 0
        for k in range(1, len(labels) + 1):
            if k == len(labels) or labels[k] != labels[k0]:
                arcs.append({"type": labels[k0], "k0": k0, "k1": k})
                k0 = k
        return arcs

    arcs = group(labels)

    # Iteratively absorb too-narrow arcs into a neighbour, then re-group so that
    # newly-adjacent same-type arcs coalesce.
    changed = True
    while changed and len(arcs) > 1:
        changed = False
        for i, arc in enumerate(arcs):
            width = (arc["k1"] - arc["k0"]) * mesh_width
            if width < min_arc_width:
                # relabel this run as its wider neighbour's type
                nb = arcs[i - 1] if i > 0 else arcs[i + 1]
                labels[arc["k0"]:arc["k1"]] = nb["type"]
                arcs = group(labels)
                changed = True
                break

    for arc in arcs:
        arc["t0"] = arc["k0"] * mesh_width
        arc["t1"] = arc["k1"] * mesh_width
    return arcs


# ========================================================================== #
# Switching machinery (control-affine a, b, F_x, sigma, phi) via CasADi AD
# ========================================================================== #
def build_switching_machinery(model):
    """Build, by CasADi AD from model.right_hand_side, the control-affine
    quantities a, b, F_x, and the switching function and its derivatives.

    Returns a dict of CasADi Functions.  Works for any control-affine
    ControlProblem (fed batch reactor, catalyst mixing, ...).
    """
    x = model.state              # MX (nx,)
    u = model.control            # MX (1,)
    k = model.params             # MX (nparams,)
    nx = model.nstates
    p = MX.sym("p", nx)

    f = model.right_hand_side    # Function([x,u,k],[xdot])
    fx = f(x, u, k)              # xdot expression (affine in u)

    b = jacobian(fx, u)          # (nx,1)  control vector field b(x)
    a = substitute(fx, u, MX(0))  # (nx,1)  drift a(x)
    Fx = jacobian(fx, x)         # (nx,nx) grad_x f, u kept SYMBOLIC

    xdot = fx
    pdot = -mtimes(Fx.T, p)      # p_dot = -(df/dx)^T p

    sigma = mtimes(p.T, b)       # switching function sigma = p^T b
    sigdot = mtimes(jacobian(sigma, x), xdot) + mtimes(jacobian(sigma, p), pdot)
    dsig_du = jacobian(sigdot, u)          # must be ~0 (order-1 singular arc)
    sigdot0 = substitute(sigdot, u, MX(0))  # u-free sigma_dot (Poisson bracket)

    sigddot = (mtimes(jacobian(sigdot0, x), xdot)
               + mtimes(jacobian(sigdot0, p), pdot))   # affine in u: A + B u
    A = substitute(sigddot, u, MX(0))
    B = jacobian(sigddot, u)
    phi = -A / B                 # generic AD singular control

    gradC = jacobian(model.final_cost_function(x), x).T   # (nx,1)

    return dict(
        nx=nx,
        f=f,
        a=Function("a", [x, k], [a]),
        b=Function("b", [x, k], [b]),
        Fx=Function("Fx", [x, u, k], [Fx]),
        sigma=Function("sigma", [x, p, k], [sigma]),
        sigdot=Function("sigdot", [x, p, k], [sigdot0]),
        sigddot=Function("sigddot", [x, p, k, u], [sigddot]),
        A=Function("A", [x, p, k], [A]),
        B=Function("B", [x, p, k], [B]),
        phi=Function("phi_AD", [x, p, k], [phi]),
        dsig_du=Function("dsig_du", [x, u, k, p], [dsig_du]),
        gradC=Function("gradC", [x], [gradC]),
    )


def _saturate_singular(phi_i, sig_i, lb, ub):
    """Feasible control on a singular arc: phi where feasible; where phi leaves
    [lb,ub] saturate per the switching-function sign (sigma<0 -> ub, >0 -> lb)."""
    if lb <= phi_i <= ub:
        return phi_i
    if sig_i < 0.0:
        return ub
    if sig_i > 0.0:
        return lb
    return min(ub, max(lb, phi_i))


# ========================================================================== #
# Ensemble singular-control reconstruction (forward state + backward adjoint)
# ========================================================================== #
def ensemble_phi_and_control(model, mach, k_samples, arcs, lb, ub, u_ws, S=8):
    """For each sample s: forward state x^(s) with the shared warm-start control,
    backward adjoint p^(s) from p^(s)(T)=gradC(x^(s)(T)).  The ENSEMBLE singular
    control is phi = -sum_s A_s / sum_s B_s (A_s=mach['A'], B_s=mach['B'] per
    sample); the ensemble switching function is sigma = sum_s p^(s).b(x^(s)).
    Returns (t, phi, u_con, sigma, C_direct, C_post) with u_con the constructed
    control (bang bound on bang arcs, phi on singular arcs) using the backward
    adjoint, and C_direct/C_post the SAA mean objective of the direct vs
    postprocessed control.
    """
    N = len(u_ws); T = model.final_time; dt = T / N; n = model.nstates
    h = dt / S; M = N * S
    f = mach["f"]; Fx = mach["Fx"]; gC = mach["gradC"]
    sigF = mach["sigma"]; AF = mach["A"]; BF = mach["B"]
    x0 = np.asarray(model.parameterized_initial_state(k_samples[0]), float)

    Xs, Ps = [], []
    for k in k_samples:
        kd = DM(k)
        X = np.zeros((M + 1, n)); X[0] = x0
        for j in range(N):
            u = float(u_ws[j])
            for r in range(S):
                x = X[j * S + r]
                a = np.asarray(f(DM(x), DM(u), kd)).ravel()
                b = np.asarray(f(DM(x + 0.5 * h * a), DM(u), kd)).ravel()
                c = np.asarray(f(DM(x + 0.5 * h * b), DM(u), kd)).ravel()
                d = np.asarray(f(DM(x + h * c), DM(u), kd)).ravel()
                X[j * S + r + 1] = x + (h / 6.0) * (a + 2 * b + 2 * c + d)
        P = np.zeros((M + 1, n)); P[M] = np.asarray(gC(DM(X[M]))).ravel()
        for i in range(M - 1, -1, -1):
            u = float(u_ws[i // S]); x = X[i + 1]; p = P[i + 1]
            J = np.asarray(Fx(DM(x), DM(u), kd)).T
            a = -(J @ p); b = -(J @ (p - 0.5 * h * a))
            c = -(J @ (p - 0.5 * h * b)); d = -(J @ (p - h * c))
            P[i] = p - (h / 6.0) * (a + 2 * b + 2 * c + d)
        Xs.append(X); Ps.append(P)

    t = np.linspace(0.0, T, M + 1)
    sig = np.zeros(M + 1); Asum = np.zeros(M + 1); Bsum = np.zeros(M + 1)
    for s, k in enumerate(k_samples):
        kd = DM(k)
        for i in range(M + 1):
            xd = DM(Xs[s][i]); pd = DM(Ps[s][i])
            sig[i] += float(sigF(xd, pd, kd))
            Asum[i] += float(AF(xd, pd, kd))
            Bsum[i] += float(BF(xd, pd, kd))
    phi = -Asum / Bsum

    ucon = np.empty(M + 1)
    for i in range(M + 1):
        arc = arcs[-1]
        for a in arcs:
            if a["t0"] <= t[i] <= a["t1"]:
                arc = a
                break
        if arc["type"] == "SING":
            ucon[i] = _saturate_singular(phi[i], sig[i], lb, ub)
        else:
            ucon[i] = lb if arc["type"] == "LOW" else ub

    # SAA mean objectives: direct control (u_ws) vs postprocessed (constructed)
    C_direct = float(np.mean([float(model.final_cost_function(DM(Xs[s][M])))
                              for s in range(len(k_samples))]))
    Cpost = []
    for k in k_samples:
        kd = DM(k)
        x = np.asarray(model.parameterized_initial_state(k), float)
        for i in range(M):
            u = float(ucon[i])
            a = np.asarray(f(DM(x), DM(u), kd)).ravel()
            b = np.asarray(f(DM(x + 0.5 * h * a), DM(u), kd)).ravel()
            c = np.asarray(f(DM(x + 0.5 * h * b), DM(u), kd)).ravel()
            d = np.asarray(f(DM(x + h * c), DM(u), kd)).ravel()
            x = x + (h / 6.0) * (a + 2 * b + 2 * c + d)
        Cpost.append(float(model.final_cost_function(DM(x))))
    C_post = float(np.mean(Cpost))
    return t, phi, ucon, sig, C_direct, C_post
