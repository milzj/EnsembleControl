import matplotlib
matplotlib.use("Agg")  # must precede any pyplot import (ensemblecontrol pulls in pyplot)
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import ensemblecontrol  # noqa: E402

from .double_integrator import DoubleIntegrator  # noqa: E402


def _solve(multiple_shooting, beta=0.0, samples=None):
    if samples is None:
        samples = [[0]]
    double_integrator = DoubleIntegrator()
    saa_problem = ensemblecontrol.SAAProblem(double_integrator, samples, beta=beta,
                                             MultipleShooting=multiple_shooting)
    w_opt, _ = saa_problem.solve()
    return saa_problem, w_opt


@pytest.mark.parametrize("multiple", [True, False])
def test_extraction_shapes(multiple):
    saa_problem, w_opt = _solve(multiple)
    plotter = ensemblecontrol.SolutionPlotter(saa_problem, w_opt)
    N = saa_problem.control_problem.nintervals
    assert plotter.tgrid.shape == (N+1,)
    assert plotter.state_mean.shape == (2, N+1)
    assert plotter.state_std.shape == (2, N+1)
    assert plotter.controls.shape == (1, N)


@pytest.mark.parametrize("multiple", [True, False])
def test_initial_state_and_zero_std(multiple):
    saa_problem, w_opt = _solve(multiple, samples=[[0]])
    plotter = ensemblecontrol.SolutionPlotter(saa_problem, w_opt)
    # DoubleIntegrator initial state is [1.0, 1.0]; both shooting types seed t=0.
    assert np.allclose(plotter.state_mean[:, 0], [1.0, 1.0])
    # A single sample carries no spread.
    assert np.allclose(plotter.state_std, 0.0)


def test_single_and_multiple_shooting_agree():
    saa_m, w_m = _solve(True)
    saa_s, w_s = _solve(False)
    plotter_m = ensemblecontrol.SolutionPlotter(saa_m, w_m)
    plotter_s = ensemblecontrol.SolutionPlotter(saa_s, w_s)
    # Same optimal control problem: trajectories and controls should match.
    assert np.allclose(plotter_m.state_mean, plotter_s.state_mean, atol=1e-2)
    assert np.allclose(plotter_m.controls, plotter_s.controls, atol=1e-2)


def test_multi_sample_shapes():
    saa_problem, w_opt = _solve(True, samples=[[0], [0], [0]])
    plotter = ensemblecontrol.SolutionPlotter(saa_problem, w_opt)
    N = saa_problem.control_problem.nintervals
    assert plotter.state_std.shape == (2, N+1)
    assert np.all(plotter.state_std >= 0.0)
    assert np.all(np.isfinite(plotter.state_mean))


@pytest.mark.parametrize("multiple", [True, False])
def test_plot_states_figure(multiple):
    saa_problem, w_opt = _solve(multiple)
    plotter = ensemblecontrol.SolutionPlotter(saa_problem, w_opt)
    fig, ax = plotter.plot_states()
    assert isinstance(fig, plt.Figure)
    assert len(ax.lines) >= saa_problem.control_problem.nstates
    plt.close("all")


@pytest.mark.parametrize("multiple", [True, False])
def test_plot_controls_figure(multiple):
    saa_problem, w_opt = _solve(multiple)
    plotter = ensemblecontrol.SolutionPlotter(saa_problem, w_opt)
    fig, ax = plotter.plot_controls()
    assert isinstance(fig, plt.Figure)
    assert len(ax.lines) >= saa_problem.control_problem.ncontrols
    plt.close("all")


def test_plot_states_reuses_given_axes():
    saa_problem, w_opt = _solve(True)
    plotter = ensemblecontrol.SolutionPlotter(saa_problem, w_opt)
    fig, ax = plt.subplots()
    returned_fig, returned_ax = plotter.plot_states(ax=ax)
    assert returned_ax is ax
    assert returned_fig is ax.figure
    assert len(ax.lines) >= 1
    plt.close("all")


def test_per_state_returns_multiple_figures():
    saa_problem, w_opt = _solve(True)
    plotter = ensemblecontrol.SolutionPlotter(saa_problem, w_opt)
    figs, axes = plotter.plot_states(per_state=True)
    assert len(figs) == saa_problem.control_problem.nstates
    assert len(axes) == saa_problem.control_problem.nstates
    plt.close("all")


@pytest.mark.parametrize("multiple", [True, False])
def test_extraction_ignores_trailing_variables(multiple):
    # CVaR (beta in (0, 1)) appends auxiliary variables to the tail of w_opt.
    # Extraction must ignore them, so appending trailing entries to a solution
    # must not change the extracted trajectories.
    saa_problem, w_opt = _solve(multiple)
    plotter = ensemblecontrol.SolutionPlotter(saa_problem, w_opt)

    w_with_tail = np.concatenate([w_opt, [3.14, 2.72, 1.61]])
    plotter_tail = ensemblecontrol.SolutionPlotter(saa_problem, w_with_tail)

    assert np.array_equal(plotter_tail.state_mean, plotter.state_mean)
    assert np.array_equal(plotter_tail.state_std, plotter.state_std)
    assert np.array_equal(plotter_tail.controls, plotter.controls)


@pytest.mark.parametrize("step", [True, False])
def test_control_step_and_line(step):
    saa_problem, w_opt = _solve(True)
    plotter = ensemblecontrol.SolutionPlotter(saa_problem, w_opt)
    fig, ax = plotter.plot_controls(step=step)
    assert len(ax.lines) >= 1
    plt.close("all")
