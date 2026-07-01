import numpy as np

__all__ = ["idx_state_control"]


def idx_state_control(nstates, ncontrols, nsamples, nintervals):
    # Index grid into the flattened multiple shooting decision vector
    # w_opt = [X0, U0, X1, U1, ..., U_{N-1}, X_N] laid out column-major
    # (order='F') as (nstates*nsamples+ncontrols, nintervals+1). idx_state
    # selects the state rows, idx_control the control rows over the intervals.
    idx = np.arange((nstates*nsamples+ncontrols)*(nintervals+1))
    idx = idx.reshape((nstates*nsamples+ncontrols, nintervals+1), order='F')
    idx_state = idx[0:nstates*nsamples, :]
    idx_control = idx[nstates*nsamples:nstates*nsamples+ncontrols+1, 0:nintervals]

    return idx_state, idx_control
