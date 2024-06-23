
import numpy as np

def idx_state_control(nstates, ncontrols, nsamples, nintervals):
  # Compute
  idx = np.arange((nstates*nsamples+ncontrols)*(nintervals+1))
  idx = idx.reshape((nstates*nsamples+ncontrols, nintervals+1), order='F')
  idx_state = idx[0:nstates*nsamples, :]
  idx_control = idx[nstates*nsamples:nstates*nsamples+ncontrols+1, 0:nintervals]

  return idx_state, idx_control

