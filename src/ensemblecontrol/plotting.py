import shutil

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .idx_state_control import idx_state_control

__all__ = ["SolutionPlotter"]

_LATEX_RCPARAMS = {
    'font.size': 8,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}',
}
_fonts_configured = False


def _configure_fonts():
    # Use LaTeX fonts when a LaTeX installation is available; otherwise keep
    # matplotlib's default mathtext rendering. Runs once per process.
    global _fonts_configured
    if _fonts_configured:
        return
    _fonts_configured = True
    if shutil.which("latex") is not None:
        plt.rcParams.update(_LATEX_RCPARAMS)


class SolutionPlotter(object):
    # Extracts the state and control trajectories from a solved SAAProblem and
    # renders them. Works for both single and multiple shooting: for multiple
    # shooting the states are read directly from w_opt, for single shooting they
    # are reconstructed by integrating the optimal controls through the ensemble
    # dynamics. Extraction happens once in the constructor (no matplotlib); the
    # plot_* methods return (fig, ax) and never call plt.show().

    def __init__(self, saa_problem, w_opt):

        self.saa_problem = saa_problem
        self.w_opt = np.asarray(w_opt).flatten()

        control_problem = saa_problem.control_problem
        self.nstates = control_problem.nstates
        self.ncontrols = control_problem.ncontrols
        self.nintervals = control_problem.nintervals
        self.mesh_width = control_problem.mesh_width
        self.alpha = control_problem.alpha
        self.nsamples = saa_problem.nsamples

        self.tgrid = np.array([self.mesh_width*k for k in range(self.nintervals+1)])

        if saa_problem.MultipleShooting:
            self._extract_multiple_shooting()
        else:
            self._extract_single_shooting()

    def _extract_multiple_shooting(self):

        nstates = self.nstates
        ncontrols = self.ncontrols
        w_opt = self.w_opt

        idx_state, idx_control = idx_state_control(nstates, ncontrols,
                                                   self.nsamples, self.nintervals)

        self.state_mean = np.empty((nstates, self.nintervals+1))
        self.state_std = np.empty((nstates, self.nintervals+1))
        for j in range(nstates):
            rows = idx_state[j::nstates]
            self.state_mean[j] = np.mean(w_opt[rows], axis=0)
            self.state_std[j] = np.std(w_opt[rows], axis=0)

        self.controls = np.empty((ncontrols, self.nintervals))
        for c in range(ncontrols):
            self.controls[c] = w_opt[idx_control[c::ncontrols]].flatten()

    def _extract_single_shooting(self):

        nstates = self.nstates
        ncontrols = self.ncontrols
        nsamples = self.nsamples
        nintervals = self.nintervals

        # Controls are stored contiguously at the front of w_opt; any CVaR
        # variables live in the tail and are dropped by the slice.
        u = self.w_opt[:nintervals*ncontrols].reshape((nintervals, ncontrols))
        self.controls = u.T

        # Reconstruct the ensemble state trajectory by rolling the controls
        # forward through the ensemble dynamics from the initial state.
        dynamics = self.saa_problem.dynamics
        X = np.array(self.saa_problem.ensemble_initial_state, dtype=float)
        ensemble = np.empty((nstates*nsamples, nintervals+1))
        ensemble[:, 0] = X
        for k in range(nintervals):
            X = np.array(dynamics(x0=X, p=u[k])['xf']).flatten()
            ensemble[:, k+1] = X

        self.state_mean = np.empty((nstates, nintervals+1))
        self.state_std = np.empty((nstates, nintervals+1))
        for j in range(nstates):
            component = ensemble[j::nstates]
            self.state_mean[j] = np.mean(component, axis=0)
            self.state_std[j] = np.std(component, axis=0)

    def plot_states(self, ax=None, state_labels=None, per_state=False, band=True,
                    annotate=True, label_prefix=None, savepath=None):

        _configure_fonts()

        if state_labels is None:
            state_labels = ["x{}".format(j) for j in range(self.nstates)]

        # One curve per provided label. When fewer labels than states are given,
        # only the labelled leading states are drawn (e.g. to skip an augmented
        # cost state).
        n_states = min(self.nstates, len(state_labels))

        if per_state:
            figs = []
            axes = []
            for j in range(n_states):
                fig, axis = plt.subplots()
                self._draw_state(axis, j, state_labels[j], band, label_prefix)
                self._finalize(axis, annotate)
                if savepath is not None:
                    fig.savefig(self._expand_savepath(savepath, state_labels[j]))
                figs.append(fig)
                axes.append(axis)
            return figs, axes

        fig, ax = self._axes(ax)
        for j in range(n_states):
            self._draw_state(ax, j, state_labels[j], band, label_prefix)
        self._finalize(ax, annotate)
        if savepath is not None:
            fig.savefig(savepath)
        return fig, ax

    def plot_controls(self, ax=None, control_labels=None, step=False,
                      annotate=True, label_prefix=None, savepath=None):

        _configure_fonts()

        if control_labels is None:
            control_labels = ["u{}".format(c) for c in range(self.ncontrols)]

        n_controls = min(self.ncontrols, len(control_labels))

        fig, ax = self._axes(ax)
        for c in range(n_controls):
            label = self._label(control_labels[c], label_prefix)
            # Controls are piecewise constant; prepend NaN so the length matches
            # tgrid. where='pre' holds control k over (t_k, t_{k+1}].
            y = np.concatenate([[np.nan], self.controls[c]])
            if step:
                ax.step(self.tgrid, y, where='pre', label=label)
            else:
                ax.plot(self.tgrid, y, '-.', label=label)
        self._finalize(ax, annotate)
        if savepath is not None:
            fig.savefig(savepath)
        return fig, ax

    def plot(self, state_labels=None, control_labels=None, band=True, step=False,
             annotate=True, label_prefix=None):

        fig_states, ax_states = self.plot_states(state_labels=state_labels, band=band,
                                                 annotate=annotate,
                                                 label_prefix=label_prefix)
        fig_controls, ax_controls = self.plot_controls(control_labels=control_labels,
                                                       step=step, annotate=annotate,
                                                       label_prefix=label_prefix)
        return (fig_states, ax_states), (fig_controls, ax_controls)

    def save(self, fig, path):
        fig.savefig(path)

    # -- helpers --------------------------------------------------------------

    def _axes(self, ax):
        if ax is None:
            return plt.subplots()
        return ax.figure, ax

    def _label(self, label, label_prefix):
        if label_prefix is None:
            return label
        return "({}) {}".format(label_prefix, label)

    def _draw_state(self, ax, j, label, band, label_prefix):
        line, = ax.plot(self.tgrid, self.state_mean[j],
                        label=self._label(label, label_prefix))
        if band:
            ax.fill_between(self.tgrid,
                            self.state_mean[j]-self.state_std[j],
                            self.state_mean[j]+self.state_std[j],
                            color=line.get_color(), alpha=0.15)

    def _finalize(self, ax, annotate):
        ax.set_xlabel(r'$t$')
        ax.grid()
        if annotate:
            handles, labels = ax.get_legend_handles_labels()
            handles.append(mpatches.Patch(color='none'))
            labels.append(r"($\alpha={}, n={}, N={}$)".format(self.alpha,
                                                              self.nintervals,
                                                              self.nsamples))
            ax.legend(handles, labels)
        else:
            ax.legend()

    def _expand_savepath(self, savepath, label):
        if "{}" in savepath:
            return savepath.format(label)
        root, dot, ext = savepath.rpartition(".")
        if dot:
            return "{}_{}.{}".format(root, label, ext)
        return "{}_{}".format(savepath, label)
