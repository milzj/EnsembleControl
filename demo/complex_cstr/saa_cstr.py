import os
import ensemblecontrol
from casadi import *
import numpy as np
from scipy.stats import qmc

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from cstr import CSTR
from idx_state_control import idx_state_control

outdir = "output"
os.makedirs(outdir, exist_ok=True)

def solve(cstr, samples):

    saa_problem = ensemblecontrol.SAAProblem(cstr, samples, MultipleShooting=True)
    w_opt, f_opt = saa_problem.solve()

    return w_opt

def plot(cstr, w_opt, samples, prefix):

    # Prepare plotting
    mesh_width = cstr.mesh_width
    nintervals = cstr.nintervals
    nstates = cstr.nstates
    ncontrols = cstr.ncontrols
    alpha = cstr.alpha
    nsamples = len(samples)

    idx_state, idx_control = idx_state_control(nstates, ncontrols, nsamples, nintervals)
    u1_opt = w_opt[idx_control[0::ncontrols]].flatten()
    u2_opt = w_opt[idx_control[1::ncontrols]].flatten()
    u3_opt = w_opt[idx_control[2::ncontrols]].flatten()
    tgrid = [mesh_width*k for k in range(nintervals+1)]

    # Controls
    plt.figure(1)
    plt.clf()
    plt.plot(tgrid, vertcat(DM.nan(1), u1_opt), '-.',color="tab:orange", label="({}) control 1".format(prefix))
    plt.plot(tgrid, vertcat(DM.nan(1), u2_opt), '--',color="tab:blue", label="({}) control 2".format(prefix))
    plt.plot(tgrid, vertcat(DM.nan(1), u3_opt), '-',color="tab:brown", label="({}) control 3".format(prefix))
    handles, labels = plt.gca().get_legend_handles_labels() # get existing handles and labels
    empty_patch = mpatches.Patch(color='none') # create a patch with no color
    handles.append(empty_patch)  # add new patches and labels to list
    labels.append(r"($\alpha={}, n={}$)".format(alpha,nintervals,nsamples))
    plt.legend(handles, labels) # apply new handles and labels to plot
    plt.xlabel(r'$t$')
    plt.grid()
    plt.savefig(outdir + "/" + prefix + "_" + "control.png")
    plt.close()

    # States
    states_idx = np.arange(nstates)

    for state_idx in states_idx:
        plt.figure(1)
        plt.clf()
        label = "x{}".format(state_idx)
        state = np.mean(w_opt[idx_state[state_idx::nstates]], axis=0)
        state_std = np.std(w_opt[idx_state[state_idx::nstates]], axis=0)
        plt.plot(tgrid, state, '-', label="({}) ".format(prefix) + label)
        plt.fill_between(tgrid, state-state_std, state+state_std, alpha=0.15)
        plt.legend()
        plt.xlabel(r'$t$')
        plt.grid()
        plt.savefig(outdir + "/" + prefix + "_" + label + ".png")
        plt.close()



if __name__ == "__main__":

    # Nominal problem
    cstr = CSTR()
    nominal_param = cstr.nominal_param
    w_opt = solve(cstr, nominal_param)
    plot(cstr, w_opt, nominal_param, "nominal")


    # Stochastic problem
    cstr = CSTR()
    nominal_param = cstr.nominal_param[0]

    # sampler
    sigma = 0.99

    nparams = len(nominal_param)
    m = 10
    sampler = qmc.Sobol(d=nparams, scramble=False)
    samples = sampler.random_base2(m=m)
    samples = qmc.scale(samples, -1.0, 1.0)
    samples = (1+sigma*samples)*nominal_param

    w_opt = solve(cstr, samples)
    plot(cstr, w_opt, samples, "stochastic")


