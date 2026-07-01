import os
import ensemblecontrol
from casadi import *
import numpy as np
from scipy.stats import qmc

import matplotlib.pyplot as plt

from fed_batch_reactor import FedBatchReactor

outdir = "output"
os.makedirs(outdir, exist_ok=True)

def solve(fed_batch_reactor, samples):

    saa_problem = ensemblecontrol.SAAProblem(fed_batch_reactor, samples, MultipleShooting=True)
    w_opt, f_opt = saa_problem.solve()

    return saa_problem, w_opt

def plot(saa_problem, w_opt, prefix):

    plotter = ensemblecontrol.SolutionPlotter(saa_problem, w_opt)
    plotter.plot_controls(control_labels=["control"], step=True, label_prefix=prefix,
                          savepath=outdir + "/" + prefix + "_control.png")
    plotter.plot_states(per_state=True, label_prefix=prefix,
                        savepath=outdir + "/" + prefix + "_{}.png")
    plt.close("all")



if __name__ == "__main__":

    # Nominal problem
    fed_batch_reactor = FedBatchReactor()
    nominal_param = fed_batch_reactor.nominal_param
    saa_problem, w_opt = solve(fed_batch_reactor, nominal_param)
    plot(saa_problem, w_opt, "nominal")


    # Risk-neutral problem
    fed_batch_reactor = FedBatchReactor()
    nominal_param = fed_batch_reactor.nominal_param[0]

    # sampler
    sigma = 0.01

    nparams = len(nominal_param)
    m = 5
    sampler = qmc.Sobol(d=nparams, scramble=False)
    samples = sampler.random_base2(m=m)
    samples = qmc.scale(samples, -1.0, 1.0)
    samples = (1+sigma*samples)*nominal_param

    saa_problem, w_opt = solve(fed_batch_reactor, samples)
    plot(saa_problem, w_opt, "risk-neutral")


