import matplotlib
matplotlib.use("Agg")  # must precede any pyplot import (ensemblecontrol pulls in pyplot)
import matplotlib.pyplot as plt  # noqa: E402

import os  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.stats import norm  # noqa: E402

import ensemblecontrol  # noqa: E402

from .double_integrator import DoubleIntegrator  # noqa: E402


# -- confidence-interval math (no solving) -----------------------------------

def test_plugin_ci_from_losses_math():
    F = np.array([-3.0, -1.0, -2.0, -4.0])
    fopt = -2.5
    ci = ensemblecontrol.plugin_ci_from_losses(F, f_opt=fopt, levels=(0.95,))
    sigma = np.sqrt(np.mean((F - fopt) ** 2))   # 1/N (population) form
    assert ci["Jhat"] == fopt
    assert ci["sigma"] == pytest.approx(sigma)
    assert ci["se"] == pytest.approx(sigma / 2.0)     # sqrt(N) = 2
    z = norm.ppf(0.975)
    band = ci["levels"][0.95]
    assert band["halfwidth"] == pytest.approx(z * sigma / 2.0)
    assert band["lo"] == pytest.approx(fopt - z * sigma / 2.0)
    assert band["hi"] == pytest.approx(fopt + z * sigma / 2.0)


def test_plugin_ci_default_center_is_mean():
    F = np.array([1.0, 2.0, 3.0])
    ci = ensemblecontrol.plugin_ci_from_losses(F, levels=(0.9,))
    assert ci["Jhat"] == pytest.approx(2.0)


def test_subsampling_ci_from_deltas_quantile():
    deltas = np.linspace(-5.0, 5.0, 11)   # sorted, m = 11
    N = 100
    ci = ensemblecontrol.subsampling_ci_from_deltas(deltas, f_opt=-2.5, N=N,
                                                    levels=(0.90,))
    band = ci["levels"][0.90]
    # beta = 0.1: quantile(0.95) at rank ceil(11*0.95)=11 -> 5; quantile(0.05)
    # at rank 1 -> -5
    assert band["quantile_hi"] == pytest.approx(5.0)
    assert band["quantile_lo"] == pytest.approx(-5.0)
    assert band["lo"] == pytest.approx(-2.5 - 5.0 / np.sqrt(N))
    assert band["hi"] == pytest.approx(-2.5 + 5.0 / np.sqrt(N))
    assert band["lo"] <= band["hi"]


# -- control extraction and terminal losses ----------------------------------

@pytest.mark.parametrize("multiple", [True, False])
def test_control_matrix_matches_plotter(multiple):
    model = DoubleIntegrator()
    saa = ensemblecontrol.SAAProblem(model, [[0], [0], [0]],
                                     MultipleShooting=multiple)
    w_opt, _ = saa.solve()
    cm = saa.control_matrix(w_opt)
    assert cm.shape == (model.nintervals, model.ncontrols)
    plotter = ensemblecontrol.SolutionPlotter(saa, w_opt)
    assert np.allclose(cm.T, plotter.controls)


def test_terminal_losses_shape_and_values():
    model = DoubleIntegrator()
    saa = ensemblecontrol.SAAProblem(model, [[0], [0]], MultipleShooting=False)
    w_opt, _ = saa.solve()
    F = ensemblecontrol.terminal_losses(saa, w_opt)
    assert F.shape == (2,)
    assert np.all(np.isfinite(F))
    # DoubleIntegrator ignores the parameter, so both scenarios give equal F,
    # and F = dot(x_T, x_T)/2 >= 0.
    assert np.allclose(F, F[0])
    assert np.all(F >= 0)


# -- subproblem / controls-only warm start -----------------------------------

@pytest.mark.parametrize("multiple", [True, False])
def test_subproblem_and_warmstart_controls(multiple):
    model = DoubleIntegrator()
    saa = ensemblecontrol.SAAProblem(model, [[0], [0], [0], [0]],
                                     MultipleShooting=multiple)
    w_opt, _ = saa.solve()
    controls = saa.control_matrix(w_opt)
    sub = saa.subproblem([0, 2])
    assert sub.nsamples == 2
    assert sub.MultipleShooting == multiple
    w0 = sub.initial_from_controls(controls)
    # controls transfer exactly; states (per-sample) stay at the subproblem's
    # own defaults and are deliberately not asserted.
    assert np.allclose(sub.control_matrix(w0), controls)


def test_subproblem_rejects_terminal_constraints():
    model = DoubleIntegrator()
    saa = ensemblecontrol.SAAProblem(
        model, [[0], [0]], MultipleShooting=False,
        terminal_constraints=[(0, lambda x: x[0], -1e9, 1e9)])
    with pytest.raises(ValueError):
        saa.subproblem([0])


# -- plug-in confidence interval end-to-end ----------------------------------

def test_plugin_confidence_interval_and_roundtrip(tmp_path):
    model = DoubleIntegrator()
    saa = ensemblecontrol.SAAProblem(model, [[0], [0], [0]],
                                     MultipleShooting=False)
    w_opt, f_opt = saa.solve()
    rec = ensemblecontrol.plugin_confidence_interval(saa, w_opt, f_opt=f_opt)
    assert rec["N"] == 3
    assert rec["F"].shape == (3,)
    assert rec["f_opt"] == pytest.approx(float(np.ravel(f_opt)[0]))
    assert 0.95 in rec["ci"]["levels"]

    path = os.path.join(tmp_path, "plugin.json")
    ensemblecontrol.save_plugin_run(rec, path)
    loaded = ensemblecontrol.load_plugin_run(path)
    assert np.allclose(loaded["results"][0]["F"], rec["F"])
    assert loaded["results"][0]["f_opt"] == pytest.approx(rec["f_opt"])
    assert os.path.exists(os.path.splitext(path)[0] + ".txt")   # secondary table


# -- subsampling confidence interval end-to-end ------------------------------

def test_subsampling_default_scipy_resolver_endtoend(tmp_path):
    model = DoubleIntegrator()
    saa = ensemblecontrol.SAAProblem(model, [[0]] * 8, MultipleShooting=False)
    w_opt, f_opt = saa.solve()
    rec = ensemblecontrol.subsampling_confidence_interval(
        saa, f_opt, b=4, m=3, rng=np.random.default_rng(0), w_opt=w_opt,
        verbose=False)
    assert rec["deltas"].shape == (3,)
    assert (rec["b"], rec["m"], rec["N"]) == (4, 3, 8)
    band = rec["ci"]["levels"][0.95]
    assert band["lo"] <= band["hi"]

    path = os.path.join(tmp_path, "sub.json")
    ensemblecontrol.save_subsampling_run(rec, path)
    loaded = ensemblecontrol.load_subsampling_run(path)
    assert np.allclose(loaded["results"][0]["deltas"], rec["deltas"])


def test_subsampling_custom_resolve_needs_no_wopt():
    model = DoubleIntegrator()
    saa = ensemblecontrol.SAAProblem(model, [[0]] * 6, MultipleShooting=True)
    _, f_opt = saa.solve()
    seen = []

    def resolve(indices):
        seen.append(len(indices))
        return f_opt   # constant resolver: every subsample returns J_hat_N*

    rec = ensemblecontrol.subsampling_confidence_interval(
        saa, f_opt, b=3, m=4, rng=np.random.default_rng(0), resolve=resolve)
    assert len(seen) == 4 and all(n == 3 for n in seen)
    assert np.allclose(rec["deltas"], 0.0)   # constant resolver -> zero deltas


def test_subsampling_default_resolver_requires_wopt():
    model = DoubleIntegrator()
    saa = ensemblecontrol.SAAProblem(model, [[0]] * 4, MultipleShooting=False)
    _, f_opt = saa.solve()
    with pytest.raises(ValueError):
        ensemblecontrol.subsampling_confidence_interval(
            saa, f_opt, b=2, m=2, rng=np.random.default_rng(0))   # no w_opt


def test_subsampling_default_resolver_rejects_multiple_shooting():
    model = DoubleIntegrator()
    saa = ensemblecontrol.SAAProblem(model, [[0]] * 4, MultipleShooting=True)
    w_opt, f_opt = saa.solve()
    with pytest.raises(ValueError):
        ensemblecontrol.subsampling_confidence_interval(
            saa, f_opt, b=2, m=2, rng=np.random.default_rng(0), w_opt=w_opt)


def test_subsampling_bad_subsample_size():
    model = DoubleIntegrator()
    saa = ensemblecontrol.SAAProblem(model, [[0]] * 4, MultipleShooting=False)
    w_opt, f_opt = saa.solve()
    with pytest.raises(ValueError):
        ensemblecontrol.subsampling_confidence_interval(
            saa, f_opt, b=4, m=2, rng=np.random.default_rng(0), w_opt=w_opt)


# -- plotting from saved raw data (headless) ---------------------------------

def test_plot_plugin_from_run_dict_multi_N():
    rng = np.random.default_rng(0)
    run = {"algorithm": "plugin", "levels": [0.90, 0.95, 0.99],
           "results": [{"N": 16, "f_opt": 1.0, "F": rng.normal(1.0, 1.0, 16)},
                       {"N": 32, "f_opt": 1.1, "F": rng.normal(1.0, 1.0, 32)}]}
    figs = ensemblecontrol.plot_plugin(run)      # in-memory, exercises scaling plot
    assert len(figs) >= 1
    plt.close("all")


def test_plot_from_saved_paths(tmp_path):
    model = DoubleIntegrator()
    saa = ensemblecontrol.SAAProblem(model, [[0]] * 4, MultipleShooting=False)
    w_opt, f_opt = saa.solve()

    rec = ensemblecontrol.plugin_confidence_interval(saa, w_opt, f_opt=f_opt)
    ppath = ensemblecontrol.save_plugin_run(rec, os.path.join(tmp_path, "p.json"))
    figs = ensemblecontrol.plot_plugin(ppath)    # from path, no recompute/solve
    assert len(figs) >= 1
    plt.close("all")

    sub = ensemblecontrol.subsampling_confidence_interval(
        saa, f_opt, b=2, m=3, rng=np.random.default_rng(0), w_opt=w_opt)
    spath = ensemblecontrol.save_subsampling_run(sub, os.path.join(tmp_path, "s.json"))
    figs = ensemblecontrol.plot_subsampling(spath)   # histogram + per-level CI plots
    assert len(figs) >= 2
    plt.close("all")


# -- CLT limit-theorem illustration ------------------------------------------

def test_clt_statistic_math():
    values = np.array([-32.0, -33.0, -31.0])
    f_ref = -32.5
    N = 64
    s = ensemblecontrol.clt_statistic(values, N, f_ref)
    assert np.allclose(s, np.sqrt(N) * (values - f_ref))


def test_clt_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    f_ref = -32.5
    values_by_N = {N: f_ref + rng.normal(0, 1.0 / np.sqrt(N), size=20)
                   for N in (32, 64, 128)}
    path = ensemblecontrol.save_clt_run(
        values_by_N, os.path.join(tmp_path, "clt.json"),
        N_ref=4096, f_ref=f_ref, q=100, meta={"model": "X"})
    run = ensemblecontrol.load_clt_run(path)
    assert run["N_ref"] == 4096 and run["q"] == 100
    assert run["f_ref"] == pytest.approx(f_ref)
    assert [r["N"] for r in run["results"]] == [32, 64, 128]
    assert np.allclose(run["results"][0]["values"], values_by_N[32])
    base = os.path.splitext(path)[0]
    assert os.path.exists(base + ".txt")               # summary
    assert os.path.exists(base + "_statistics.csv")    # per-N statistic columns


def test_plot_clt_from_dict_and_path(tmp_path):
    rng = np.random.default_rng(1)
    f_ref = 1.0
    values_by_N = {N: f_ref + rng.normal(0, 1.0 / np.sqrt(N), size=30)
                   for N in (32, 64, 128)}
    run = {"algorithm": "clt", "N_ref": 4096, "f_ref": f_ref, "q": 50,
           "results": [{"N": N, "values": values_by_N[N]} for N in (32, 64, 128)]}
    figs = ensemblecontrol.plot_clt(run)               # in-memory, no solve
    assert len(figs) >= 4                               # 3 per-N + combined
    plt.close("all")

    path = ensemblecontrol.save_clt_run(
        values_by_N, os.path.join(tmp_path, "clt.json"), N_ref=4096, f_ref=f_ref)
    saved = ensemblecontrol.plot_clt(path, outdir=str(tmp_path))   # from JSON path
    assert any(p.endswith("_clt_all.png") for p in saved)
    plt.close("all")
