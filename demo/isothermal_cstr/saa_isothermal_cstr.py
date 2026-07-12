import os
import ensemblecontrol
from casadi import *
import numpy as np
from scipy.stats import qmc

import matplotlib.pyplot as plt

from isothermal_cstr import IsothermalCSTR

outdir = "output"
os.makedirs(outdir, exist_ok=True)

def solve(problem, samples, beta=0.0):

    saa_problem = ensemblecontrol.SAAProblem(problem, samples, beta=beta,
                                             MultipleShooting=True)
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
    problem = IsothermalCSTR()
    saa_problem, w_opt = solve(problem, problem.nominal_param, beta=0.0)
    plot(saa_problem, w_opt, "nominal")

    # Sample the uncertain parameters (+/- 10 % around nominal)
    problem = IsothermalCSTR()
    nominal_param = problem.nominal_param[0]

    sigma = 0.10
    nparams = len(nominal_param)
    m = 5
    sampler = qmc.Sobol(d=nparams, scramble=False)
    samples = sampler.random_base2(m=m)
    samples = qmc.scale(samples, -1.0, 1.0)
    samples = (1 + sigma*samples)*nominal_param

    # Risk-neutral problem
    saa_problem, w_opt = solve(problem, samples, beta=0.0)
    plot(saa_problem, w_opt, "risk-neutral")

    # Risk-averse problem (CVaR)
    saa_problem, w_opt = solve(problem, samples, beta=0.9)
    plot(saa_problem, w_opt, "cvar")
