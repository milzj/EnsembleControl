import os
import ensemblecontrol
from casadi import *
import numpy as np
from scipy.stats import qmc

import matplotlib.pyplot as plt

from cstr import CSTR

outdir = "output"
os.makedirs(outdir, exist_ok=True)

def solve(cstr, samples):

    saa_problem = ensemblecontrol.SAAProblem(cstr, samples, MultipleShooting=True)
    w_opt, f_opt = saa_problem.solve()

    return saa_problem, w_opt

def plot(saa_problem, w_opt, prefix):

    plotter = ensemblecontrol.SolutionPlotter(saa_problem, w_opt)
    plotter.plot_controls(control_labels=["control 1", "control 2", "control 3"],
                          label_prefix=prefix,
                          savepath=outdir + "/" + prefix + "_control.png")
    plotter.plot_states(per_state=True, label_prefix=prefix,
                        savepath=outdir + "/" + prefix + "_{}.png")
    plt.close("all")



if __name__ == "__main__":

    # Nominal problem
    cstr = CSTR()
    nominal_param = cstr.nominal_param
    saa_problem, w_opt = solve(cstr, nominal_param)
    plot(saa_problem, w_opt, "nominal")


    # Risk-neutral problem
    cstr = CSTR()
    nominal_param = cstr.nominal_param[0]

    # sampler
    sigma = 0.99

    nparams = len(nominal_param)
    m = 5
    sampler = qmc.Sobol(d=nparams, scramble=False)
    samples = sampler.random_base2(m=m)
    samples = qmc.scale(samples, -1.0, 1.0)
    samples = (1+sigma*samples)*nominal_param

    saa_problem, w_opt = solve(cstr, samples)
    plot(saa_problem, w_opt, "risk-neutral")


