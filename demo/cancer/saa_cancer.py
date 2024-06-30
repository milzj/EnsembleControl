import ensemblecontrol
from casadi import *
import numpy as np
from scipy.stats import qmc

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.rcParams.update({
    'font.size': 8,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})


from cancer import Cancer


cancer = Cancer()
nominal_param = cancer.nominal_param[0]


# sampler
sigma = 0.01
nparams = len(nominal_param)
m = 10
sampler = qmc.Sobol(d=nparams, scramble=False)
samples = sampler.random_base2(m=m)
samples = qmc.scale(samples, -1.0, 1.0)
samples = (1+sigma*samples)*nominal_param

#samples = cancer.nominal_param

saa_problem = ensemblecontrol.SAAProblem(cancer, samples, MultipleShooting=True)

w_opt, f_opt = saa_problem.solve()


# Prepare plotting

mesh_width = cancer.mesh_width
nintervals = cancer.nintervals
nstates = cancer.nstates
ncontrols = cancer.ncontrols
alpha = cancer.alpha
nsamples = len(samples)

def idx_state_control(nstates, ncontrols, nsamples, nintervals):

  idx = np.arange((nstates*nsamples+ncontrols)*(nintervals+1))
  idx = idx.reshape((nstates*nsamples+ncontrols, nintervals+1), order='F')
  idx_state = idx[0:nstates*nsamples, :]
  idx_control = idx[nstates*nsamples:nstates*nsamples+ncontrols+1, 0:nintervals]

  return idx_state, idx_control


idx_state, idx_control = idx_state_control(nstates, ncontrols, nsamples, nintervals)

x1_opt = np.mean(w_opt[idx_state[0::nstates]], axis=0)
x1_opt_std = np.std(w_opt[idx_state[0::nstates]], axis=0)

u1_opt = w_opt[idx_control[0::ncontrols]].flatten()

tgrid = [mesh_width*k for k in range(nintervals+1)]

# Controls


plt.figure(1)
plt.clf()
plt.plot(tgrid, vertcat(DM.nan(1), u1_opt), '-.',color="tab:orange", label=r"$u_1^*(t)$")

handles, labels = plt.gca().get_legend_handles_labels() # get existing handles and labels
empty_patch = mpatches.Patch(color='none') # create a patch with no color

handles.append(empty_patch)  # add new patches and labels to list
labels.append(r"($\alpha={}, n={}, N={}$)".format(alpha,nintervals,nsamples))

plt.legend(handles, labels) # apply new handles and labels to plot
plt.xlabel(r'$t$')
plt.grid()
plt.savefig("controls.png")


# States

plt.figure(1)
plt.clf()
plt.plot(tgrid, x1_opt, '--', label=r"$\mathbb{E}[x_1^*(t,\xi)]$", color="tab:blue")
plt.fill_between(tgrid,x1_opt-x1_opt_std, x1_opt+x1_opt_std, color="tab:blue", alpha=0.15, label=r"$\mathbb{E}[x_1^*(t,\xi)]\pm \mathrm{std}[x_1^*(t,\xi)]$")
plt.legend()
plt.xlabel(r'$t$')
plt.grid()
plt.savefig("states.png")
