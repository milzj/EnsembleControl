import ensemblecontrol
from casadi import *
import numpy as np
from scipy.stats import qmc

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from idx_state_control import idx_state_control
from catalyst_mixing import CatalystMixing

catalyst_mixing = CatalystMixing()

sigma = 0.99
m = 6
sampler = qmc.Sobol(d=3, scramble=False)
samples = sampler.random_base2(m=m)
samples = qmc.scale(samples, -1., 1.)
samples = (1+sigma*samples)*catalyst_mixing.nominal_param[0]


saa_problem = ensemblecontrol.SAAProblem(catalyst_mixing, samples)

w_opt, f_opt = saa_problem.solve()

# Prepare plotting
mesh_width = catalyst_mixing.mesh_width
nintervals = catalyst_mixing.nintervals
nstates = catalyst_mixing.nstates
ncontrols = catalyst_mixing.ncontrols
alpha = catalyst_mixing.alpha
nsamples = len(samples)

idx_state, idx_control = idx_state_control(nstates, ncontrols, nsamples, nintervals)

x1_opt = np.mean(w_opt[idx_state[0::nstates]], axis=0)
x1_opt_std = np.std(w_opt[idx_state[0::nstates]], axis=0)

x2_opt = np.mean(w_opt[idx_state[1::nstates]], axis=0)
x2_opt_std = np.std(w_opt[idx_state[1::nstates]], axis=0)

u1_opt = w_opt[idx_control[0::ncontrols]].flatten()

tgrid = [mesh_width*k for k in range(nintervals+1)]

# Controls
plt.figure(1)
plt.clf()
plt.step(tgrid, vertcat(DM.nan(1), u1_opt), '-.',color="tab:orange", label=r"$u_1^*(t)$")

handles, labels = plt.gca().get_legend_handles_labels() # get existing handles and labels
empty_patch = mpatches.Patch(color='none') # create a patch with no color

handles.append(empty_patch)  # add new patches and labels to list
labels.append(r"($\alpha={}, n={}, N={}$)".format(alpha,nintervals,nsamples))

plt.legend(handles, labels) # apply new handles and labels to plot
plt.xlabel(r'$t$')
plt.grid()
plt.savefig("controls.pdf")


# States
plt.figure(1)
plt.clf()
plt.plot(tgrid, x1_opt, '--', label=r"$\mathbb{E}[x_1^*(t,\xi)]$", color="tab:blue")
plt.plot(tgrid, x2_opt, '-', label=r"$\mathbb{E}[x_2^*(t,\xi)]$", color="tab:orange")
plt.fill_between(tgrid,x1_opt-x1_opt_std, x1_opt+x1_opt_std, color="tab:blue", alpha=0.15, label=r"$\mathbb{E}[x_1^*(t,\xi)]\pm \mathrm{std}[x_1^*(t,\xi)]$")
plt.fill_between(tgrid,x2_opt-x2_opt_std, x2_opt+x2_opt_std, color="tab:orange", alpha=0.15, label=r"$\mathbb{E}[x_2^*(t,\xi)]\pm \mathrm{std}[x_2^*(t,\xi)]$")
plt.legend()
plt.xlabel(r'$t$')
plt.grid()
plt.savefig("states.pdf")
