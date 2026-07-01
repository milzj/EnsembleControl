import os
import ensemblecontrol
from casadi import *
import numpy as np
from scipy.stats import qmc

from cubic_oscillator import CubicOscillator


cubic_oscillator = CubicOscillator()
nominal_param = cubic_oscillator.nominal_param[0]


# sampler
nparams = len(nominal_param)
m = 5
sampler = qmc.Sobol(d=nparams, scramble=False)
samples = 2.0*np.pi*sampler.random_base2(m=m)

#samples = cubic_oscillator.nominal_param

saa_problem = ensemblecontrol.SAAProblem(cubic_oscillator, samples, MultipleShooting=True)

w_opt, f_opt = saa_problem.solve()


# Plot states and controls
outdir = "output"
os.makedirs(outdir, exist_ok=True)
plotter = ensemblecontrol.SolutionPlotter(saa_problem, w_opt)
plotter.plot_controls(control_labels=[r"$u_1^*(t)$", r"$u_2^*(t)$"],
                      savepath=outdir + "/controls.pdf")
plotter.plot_states(state_labels=[r"$\mathbb{E}[x_1^*(t,\xi)]$", r"$\mathbb{E}[x_2^*(t,\xi)]$"],
                    savepath=outdir + "/states.pdf")
