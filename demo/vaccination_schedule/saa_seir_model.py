import os
import ensemblecontrol
from casadi import *
import numpy as np
from scipy.stats import qmc

import matplotlib.pyplot as plt

from seir_model import SEIRModel

outdir = "output"
os.makedirs(outdir, exist_ok=True)

def solve(seir_model, samples, beta=0.0):

    saa_problem = ensemblecontrol.SAAProblem(seir_model, samples, beta=beta, MultipleShooting=True)
    w_opt, f_opt = saa_problem.solve()

    return saa_problem, w_opt

def plot(saa_problem, w_opt, prefix):

    labels = ["susceptible", "exposed", "infectious", "recovered", "total population"]
    plotter = ensemblecontrol.SolutionPlotter(saa_problem, w_opt)
    plotter.plot_controls(control_labels=["vaccination rate"], label_prefix=prefix,
                          savepath=outdir + "/" + prefix + "_vaccination_rate.png")
    plotter.plot_states(per_state=True, state_labels=labels, label_prefix=prefix,
                        savepath=outdir + "/" + prefix + "_{}.png")
    plt.close("all")



if __name__ == "__main__":

    beta = 0.9
    # Nominal problem
    seir_model = SEIRModel()
    nominal_param = seir_model.nominal_param
    saa_problem, w_opt = solve(seir_model, nominal_param)
    plot(saa_problem, w_opt, "nominal")


    # Risk-averse problem
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

    saa_problem, w_opt = solve(seir_model, samples, beta=beta)
    plot(saa_problem, w_opt, "risk-averse")


