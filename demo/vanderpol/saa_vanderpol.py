import os
import ensemblecontrol
from casadi import *
import numpy as np
from scipy.stats import qmc

from vanderpol import VanDerPol


vanderpol = VanDerPol()
nominal_param = vanderpol.nominal_param[0]


# sampler
sigma = 0.1
nparams = len(nominal_param)
m = 5
sampler = qmc.Sobol(d=nparams, scramble=False)
samples = sampler.random_base2(m=m)
samples = qmc.scale(samples, -1.0, 1.0)
samples = (1+sigma*samples)*nominal_param

#samples = vanderpol.nominal_param

saa_problem = ensemblecontrol.SAAProblem(vanderpol, samples, MultipleShooting=True)

w_opt, f_opt = saa_problem.solve()


# Plot states and controls
outdir = "output"
os.makedirs(outdir, exist_ok=True)
plotter = ensemblecontrol.SolutionPlotter(saa_problem, w_opt)
plotter.plot_controls(control_labels=[r"$u_1^*(t)$"], savepath=outdir + "/controls.pdf")
plotter.plot_states(state_labels=[r"$\mathbb{E}[x_1^*(t,\xi)]$", r"$\mathbb{E}[x_2^*(t,\xi)]$"],
                    savepath=outdir + "/states.pdf")
