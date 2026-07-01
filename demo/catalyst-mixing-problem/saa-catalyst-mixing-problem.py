import os
import ensemblecontrol
from casadi import *
import numpy as np
from scipy.stats import qmc

import matplotlib.pyplot as plt

from catalyst_mixing import CatalystMixing

catalyst_mixing = CatalystMixing()

sigma = 0.99
m = 5
sampler = qmc.Sobol(d=3, scramble=False)
samples = sampler.random_base2(m=m)
samples = qmc.scale(samples, -1., 1.)
samples = (1+sigma*samples)*catalyst_mixing.nominal_param[0]


saa_problem = ensemblecontrol.SAAProblem(catalyst_mixing, samples)

w_opt, f_opt = saa_problem.solve()

# Plot states and controls
outdir = "output"
os.makedirs(outdir, exist_ok=True)
plotter = ensemblecontrol.SolutionPlotter(saa_problem, w_opt)
plotter.plot_controls(control_labels=[r"$u_1^*(t)$"], step=True,
                      savepath=outdir + "/controls.pdf")
plotter.plot_states(state_labels=[r"$\mathbb{E}[x_1^*(t,\xi)]$", r"$\mathbb{E}[x_2^*(t,\xi)]$"],
                    savepath=outdir + "/states.pdf")
