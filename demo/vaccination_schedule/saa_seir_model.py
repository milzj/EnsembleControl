import os
import ensemblecontrol
from casadi import *
import numpy as np
from scipy.stats import qmc

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from seir_model import SEIRModel
from idx_state_control import idx_state_control

outdir = "output"
os.makedirs(outdir, exist_ok=True)

def solve(seir_model, samples, beta=0.0):

    saa_problem = ensemblecontrol.SAAProblem(seir_model, samples, beta=beta, MultipleShooting=True)
    w_opt, f_opt = saa_problem.solve()

    return w_opt

def plot(seir_model, w_opt, samples, prefix):

    # Prepare plotting
    mesh_width = seir_model.mesh_width
    nintervals = seir_model.nintervals
    nstates = seir_model.nstates
    ncontrols = seir_model.ncontrols
    alpha = seir_model.alpha
    nsamples = len(samples)

    idx_state, idx_control = idx_state_control(nstates, ncontrols, nsamples, nintervals)
    u1_opt = w_opt[idx_control[0::ncontrols]].flatten()
    tgrid = [mesh_width*k for k in range(nintervals+1)]

    # Controls
    plt.figure(1)
    plt.clf()
    plt.plot(tgrid, vertcat(DM.nan(1), u1_opt), '-.',color="tab:orange", label="({}) vaccination rate".format(prefix))
    handles, labels = plt.gca().get_legend_handles_labels() # get existing handles and labels
    empty_patch = mpatches.Patch(color='none') # create a patch with no color
    handles.append(empty_patch)  # add new patches and labels to list
    labels.append(r"($\alpha={}, n={}, N={}$)".format(alpha,nintervals,nsamples))
    plt.legend(handles, labels) # apply new handles and labels to plot
    plt.xlabel(r'$t$')
    plt.grid()
    plt.savefig(outdir + "/" + prefix + "_" + "vaccination_rate.png")
    plt.close()

    # States
    labels = ["susceptible", "exposed", "infectious", "recovered", "total population"]
    states_idx = np.arange(len(labels))

    for state_idx, label in zip(states_idx, labels):
        plt.figure(1)
        plt.clf()
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

    beta = 0.9
    # Nominal problem
    seir_model = SEIRModel()
    nominal_param = seir_model.nominal_param
    w_opt = solve(seir_model, nominal_param, beta=beta)
    plot(seir_model, w_opt, nominal_param, "nominal")


    # Stochastic problem
    seir_model = SEIRModel()
    nominal_param = seir_model.nominal_param[0]

    # sampler
    sigma = 0.05

    nparams = len(nominal_param)
    m = 5
    sampler = qmc.Sobol(d=nparams, scramble=False)
    samples = sampler.random_base2(m=m)
    samples = qmc.scale(samples, -1.0, 1.0)
    samples = (1+sigma*samples)*nominal_param

    w_opt = solve(seir_model, samples, beta=beta)
    plot(seir_model, w_opt, samples, "stochastic")


