import numpy as np

from . import norm_vec

def canonical_criticality_measure(u_vec, grad_vec, lb_vec, ub_vec, mesh_width):
    # Criticality measure for min f(u) subject to lb_vec <= u <= ub_vec
    w_vec = u_vec-grad_vec
    proj_w = np.clip(w_vec, lb_vec, ub_vec)
    return norm_vec(u_vec-proj_w, mesh_width)
