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


from seir_model import SEIRModel


seir_model = SEIRModel()
nominal_param = seir_model.nominal_param[0]

# sampler
sigma = 0.1
nparams = len(nominal_param)
m = 10
sampler = qmc.Sobol(d=nparams, scramble=False)
samples = sampler.random_base2(m=m)
samples = qmc.scale(samples, -1.0, 1.0)
samples = (1+sigma*samples)*nominal_param

#samples = seir_model.nominal_param

saa_problem = ensemblecontrol.SAAProblem(seir_model, samples, MultipleShooting=True)

w_opt, f_opt = saa_problem.solve()


# Prepare plotting

mesh_width = seir_model.mesh_width
nintervals = seir_model.nintervals
nstates = seir_model.nstates
ncontrols = seir_model.ncontrols
alpha = seir_model.alpha
nsamples = len(samples)

def idx_state_control(nstates, ncontrols, nsamples, nintervals):

  idx = np.arange((nstates*nsamples+ncontrols)*(nintervals+1))
  idx = idx.reshape((nstates*nsamples+ncontrols, nintervals+1), order='F')
  idx_state = idx[0:nstates*nsamples, :]
  idx_control = idx[nstates*nsamples:nstates*nsamples+ncontrols+1, 0:nintervals]

  return idx_state, idx_control

idx_state, idx_control = idx_state_control(nstates, ncontrols, nsamples, nintervals)

susceptible_opt = np.mean(w_opt[idx_state[0::nstates]], axis=0)
susceptible_opt_std = np.std(w_opt[idx_state[0::nstates]], axis=0)
exposed_opt = np.mean(w_opt[idx_state[1::nstates]], axis=0)
exposed_opt_std = np.std(w_opt[idx_state[1::nstates]], axis=0)
infectious_opt = np.mean(w_opt[idx_state[2::nstates]], axis=0)
infectious_opt_std = np.std(w_opt[idx_state[2::nstates]], axis=0)
recovered_opt = np.mean(w_opt[idx_state[3::nstates]], axis=0)
recovered_opt_std = np.std(w_opt[idx_state[3::nstates]], axis=0)
total_population_opt = np.mean(w_opt[idx_state[4::nstates]], axis=0)
total_population_opt_std = np.std(w_opt[idx_state[4::nstates]], axis=0)

u1_opt = w_opt[idx_control[0::ncontrols]].flatten()

tgrid = [mesh_width*k for k in range(nintervals+1)]

# Controls


plt.figure(1)
plt.clf()
plt.plot(tgrid, vertcat(DM.nan(1), u1_opt), '-.',color="tab:orange", label="vaccination rate")
handles, labels = plt.gca().get_legend_handles_labels() # get existing handles and labels
empty_patch = mpatches.Patch(color='none') # create a patch with no color
handles.append(empty_patch)  # add new patches and labels to list
labels.append(r"($\alpha={}, n={}, N={}$)".format(alpha,nintervals,nsamples))
plt.legend(handles, labels) # apply new handles and labels to plot
plt.xlabel(r'$t$')
plt.grid()
plt.savefig("vaccination_rate.pdf")
plt.close()


states = [susceptible_opt, exposed_opt, infectious_opt, recovered_opt, total_population_opt]
states_std = [susceptible_opt_std, exposed_opt_std, infectious_opt_std, recovered_opt_std, total_population_opt_std]
labels = ["susceptible", "exposed", "infectious", "recovered", "total population"]

for state, state_std, label in zip(states, states_std, labels):
    plt.figure(1)
    plt.clf()
    plt.plot(tgrid, state, '-', label=label)
    plt.fill_between(tgrid, state-state_std, state+state_std, alpha=0.15)
    plt.legend()
    plt.xlabel(r'$t$')
    plt.grid()
    plt.savefig(label + ".pdf")
    plt.close()

