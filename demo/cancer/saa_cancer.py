import os
import ensemblecontrol
from casadi import *
import numpy as np
from scipy.stats import qmc

from cancer import Cancer


cancer = Cancer()
nominal_param = cancer.nominal_param[0]


# sampler
sigma = 0.01
nparams = len(nominal_param)
m = 5
sampler = qmc.Sobol(d=nparams, scramble=False)
samples = sampler.random_base2(m=m)
samples = qmc.scale(samples, -1.0, 1.0)
samples = (1+sigma*samples)*nominal_param

#samples = cancer.nominal_param

saa_problem = ensemblecontrol.SAAProblem(cancer, samples, MultipleShooting=True)

w_opt, f_opt = saa_problem.solve()


# Plot states and controls
outdir = "output"
os.makedirs(outdir, exist_ok=True)
plotter = ensemblecontrol.SolutionPlotter(saa_problem, w_opt)
plotter.plot_controls(control_labels=[r"$u_1^*(t)$"], savepath=outdir + "/controls.png")
plotter.plot_states(state_labels=[r"$\mathbb{E}[x_1^*(t,\xi)]$"], savepath=outdir + "/states.png")
