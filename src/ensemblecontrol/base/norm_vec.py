import numpy as np

def norm_vec(u_vec, mesh_width):
    # Compute L2 norm
    return np.sqrt(mesh_width*np.dot(u_vec,u_vec))
