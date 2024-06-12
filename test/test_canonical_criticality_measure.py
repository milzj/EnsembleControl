import ensemblecontrol.base

import numpy as np

def test_canonical_criticality_measure():

    n = 10

    lb_vec = -np.ones(n)
    ub_vec = np.ones(n)
    grad_vec = np.random.randn(n)
    u_vec = np.zeros(n)


    idx = grad_vec > 0.0
    u_vec[idx] = -1.0

    idx = grad_vec < 0.0
    u_vec[idx] = 1.0

    mesh_width = 1.0

    cm = ensemblecontrol.base.canonical_criticality_measure(u_vec,
                                                        grad_vec,
                                                        lb_vec,
                                                        ub_vec,
                                                        mesh_width)
    assert cm == 0.0
