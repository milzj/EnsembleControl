import numpy as np

from . import norm_vec

def canonical_criticality_measure(u_vec, grad_vec, lb_vec, ub_vec, mesh_width):
    # Criticality measure for min f(u) subject to lb_vec <= u <= ub_vec, in the
    # discretized L2 inner product <u,v> = mesh_width * dot(u,v) (mass M = h*I for
    # piecewise-constant controls on a uniform grid). The projected-gradient step
    # uses the Riesz gradient M^{-1} grad = grad/mesh_width, not the Euclidean
    # grad; the box projection is componentwise (M diagonal) and the residual is
    # measured in the same L2 norm.
    w_vec = u_vec-grad_vec/mesh_width
    proj_w = np.clip(w_vec, lb_vec, ub_vec)
    return norm_vec(u_vec-proj_w, mesh_width)
