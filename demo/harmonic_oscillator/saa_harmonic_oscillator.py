import os
import ensemblecontrol
from casadi import *
import numpy as np
from scipy.stats import qmc

import matplotlib.pyplot as plt

from harmonic_oscillator import HarmonicOscillator

harmonic_oscillator = HarmonicOscillator()

sampler = qmc.Sobol(d=1, scramble=False)
samples = 2.0*np.pi*sampler.random_base2(m=5)

saa_problem = ensemblecontrol.SAAProblem(harmonic_oscillator, samples)

w_opt, f_opt = saa_problem.solve()

# Plot states and controls
outdir = "output"
os.makedirs(outdir, exist_ok=True)
plotter = ensemblecontrol.SolutionPlotter(saa_problem, w_opt)
plotter.plot_controls(control_labels=[r"$u_1^*(t)$", r"$u_2^*(t)$"],
                      savepath=outdir + "/controls.pdf")
plotter.plot_states(state_labels=[r"$\mathbb{E}[x_1^*(t,\xi)]$", r"$\mathbb{E}[x_2^*(t,\xi)]$"],
                    savepath=outdir + "/states.pdf")
