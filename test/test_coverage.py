import matplotlib
matplotlib.use("Agg")  # must precede any pyplot import (ensemblecontrol pulls in pyplot)

import os  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import ensemblecontrol  # noqa: E402
from ensemblecontrol import (coverage_from_indicators, coverage_latex_table,  # noqa: E402
                             coverage_study, probability_lower_bound,
                             save_coverage_run, load_coverage_run)

from .double_integrator import DoubleIntegrator  # noqa: E402


# -- fakes for the orchestration tests (no solving) --------------------------

def fake_solve(samples, w0=None, inner_serial=False):
    # No optimization: the "optimal value" is a deterministic function of the draw
    # so the coverage indicators vary and are order-sensitive.
    return None, np.zeros(3), float(np.asarray(samples).sum())


def fake_ci_of(saa, w_opt, f_opt):
    # Two levels with different half-widths centered at f_opt -> the levels' rows
    # differ and coverage varies across replications.
    return {"levels": {0.90: {"lo": f_opt - 0.3, "hi": f_opt + 0.3},
                       0.95: {"lo": f_opt - 1.0, "hi": f_opt + 1.0}}}


def synthetic_study():
    return {
        "sample_sizes": [32, 64],
        "levels": [0.90, 0.95],
        "n_ref": 100, "f_ref": -1.25, "q": 50, "R": 10,
        "w_ref": np.zeros(50),
        "indicators_by_N": {
            32: {0.90: np.array([True] * 9 + [False]),
                 0.95: np.array([True] * 10)},
            64: {0.90: np.array([True] * 7 + [False] * 3),
                 0.95: np.array([True] * 9 + [False])},
        },
    }


# -- coverage_from_indicators (pure) -----------------------------------------

def test_coverage_from_indicators_math():
    indicators = {0.90: np.array([True] * 7 + [False] * 3),
                  0.95: np.array([True] * 9 + [False])}
    agg = coverage_from_indicators(indicators, deltas=(0.05, 0.1))
    assert agg[0.90]["L"] == 7 and agg[0.90]["R"] == 10
    assert agg[0.90]["coverage"] == pytest.approx(0.7)
    for delta in (0.05, 0.1):
        assert agg[0.90]["lower_bounds"][delta] == pytest.approx(
            probability_lower_bound(10, 7, delta))


def test_coverage_from_indicators_all_and_none():
    R = 8
    allcov = coverage_from_indicators({0.95: np.ones(R, dtype=bool)})
    assert allcov[0.95]["coverage"] == 1.0
    assert allcov[0.95]["lower_bounds"][0.05] == pytest.approx(0.05 ** (1 / R))
    none = coverage_from_indicators({0.95: np.zeros(R, dtype=bool)})
    assert none[0.95]["coverage"] == 0.0
    assert none[0.95]["lower_bounds"][0.05] == 0.0


# -- coverage_latex_table (pure) ---------------------------------------------

def test_coverage_latex_table_content():
    tex = coverage_latex_table(synthetic_study(), deltas=(0.05,),
                               caption="My caption", label="tab:cov")
    assert "\\toprule" in tex and "\\bottomrule" in tex and "\\midrule" in tex
    assert "My caption" in tex and "tab:cov" in tex
    assert "$1-\\alpha = 0.90$" in tex and "$1-\\alpha = 0.95$" in tex
    # N=32, level 0.90 -> coverage 9/10 = 0.900 and its lower bound at delta=0.05
    assert "0.900" in tex
    bound = probability_lower_bound(10, 9, 0.05)
    assert "{:.3f}".format(bound) in tex


@pytest.mark.parametrize("delta,sub", [
    (1e-6, "10^{-6}"), (1e-5, "10^{-5}"), (1e-3, "10^{-3}"),
    (0.1, "10^{-1}"), (0.05, "0.05"), (0.025, "0.025")])
def test_coverage_latex_table_delta_subscript(delta, sub):
    # powers of ten render as 10^{k} (any exponent); other deltas stay decimal
    tex = coverage_latex_table(synthetic_study(), deltas=(delta,))
    assert ("$\\underline{p}_{%s}$" % sub) in tex
    assert "1e-0" not in tex          # never leak python's e-notation


def test_coverage_latex_table_mixed_deltas_render_each():
    tex = coverage_latex_table(synthetic_study(), deltas=(0.05, 1e-6))
    assert "$\\underline{p}_{0.05}$" in tex
    assert "$\\underline{p}_{10^{-6}}$" in tex


def test_coverage_latex_table_multiple_deltas_adds_columns():
    one = coverage_latex_table(synthetic_study(), deltas=(0.05,))
    two = coverage_latex_table(synthetic_study(), deltas=(0.05, 0.1))
    # each level gains one lower-bound column per extra delta
    assert two.count("underline") == one.count("underline") + 2


# -- save / load round-trip --------------------------------------------------

def test_save_load_roundtrip_and_table(tmp_path):
    study = synthetic_study()
    path = os.path.join(tmp_path, "coverage.json")
    save_coverage_run(study, path, meta={"model": "synthetic"})
    assert os.path.exists(os.path.splitext(path)[0] + ".txt")

    loaded = load_coverage_run(path)
    assert loaded["f_ref"] == study["f_ref"]
    assert loaded["R"] == study["R"]
    assert list(loaded["levels"]) == study["levels"]
    for res in loaded["results"]:
        N = res["N"]
        for lvl in loaded["levels"]:
            assert res["indicators"][lvl].dtype == bool
            assert np.array_equal(res["indicators"][lvl],
                                  study["indicators_by_N"][N][lvl])
    # the table is identical whether built from the path, the loaded run, or memory
    assert coverage_latex_table(path) == coverage_latex_table(loaded)
    assert coverage_latex_table(path) == coverage_latex_table(study)


# -- coverage_study orchestration (fake solve, no solving) -------------------

def test_coverage_study_workers_equivalence():
    def run(workers):
        root = ensemblecontrol.UniformSampler(0.0, 1.0, method="mc", seed=5)
        return coverage_study(root, fake_solve, sample_sizes=(4, 6), R=20, n_ref=8,
                              levels=(0.90, 0.95), ci_of=fake_ci_of, workers=workers)

    base = run(1)
    for lvl in (0.90, 0.95):
        for N in (4, 6):
            ind = base["indicators_by_N"][N][lvl]
            assert ind.shape == (20,) and ind.dtype == bool
            assert ind.std() > 0            # nonconstant -> the test is meaningful
    for workers in ("auto", 2, 4):
        other = run(workers)
        assert other["f_ref"] == base["f_ref"]
        for N in (4, 6):
            for lvl in (0.90, 0.95):
                assert np.array_equal(other["indicators_by_N"][N][lvl],
                                      base["indicators_by_N"][N][lvl])


def test_coverage_study_uses_common_random_numbers():
    # Within each replicate the size-N training sample must be the nested PREFIX of
    # the size-max sample (common random numbers across N -- the canonical SAA
    # construction). A recording fake solve captures the samples each solve receives.
    seen = {}

    def rec_solve(samples, w0=None, inner_serial=False):
        s = np.asarray(samples, dtype=float)
        seen.setdefault(s.shape[0], []).append(s.copy())
        return None, np.zeros(3), float(s.sum())

    root = ensemblecontrol.UniformSampler(0.0, 1.0, method="mc", seed=7)
    coverage_study(root, rec_solve, sample_sizes=(6, 10), R=5, n_ref=8,
                   levels=(0.90, 0.95), ci_of=fake_ci_of, workers=1)

    for r in range(5):
        # replicate r's size-6 problem is the first 6 scenarios of its size-10 problem
        assert np.allclose(seen[6][r], seen[10][r][:6])
    # independent replicates draw different sequences
    assert not np.allclose(seen[10][0], seen[10][1])
    # n_ref=8 differs from every training size -> the reference solve stays on its
    # own key and is NOT one of the nested training samples (independent target)
    assert len(seen[8]) == 1


# -- coverage_study end-to-end with the real plug-in CI (tiny) ---------------

def _double_integrator_solve():
    model = DoubleIntegrator()
    model.nintervals = 20                    # small mesh -> fast solves
    return ensemblecontrol.make_scipy_solve(model, tol=1e-6)


def test_coverage_study_default_plugin_endtoend():
    solve = _double_integrator_solve()
    sampler = ensemblecontrol.UniformSampler(0.0, 1.0, method="mc", seed=0)
    study = coverage_study(sampler, solve, sample_sizes=(3, 4), R=4, n_ref=6,
                           workers=1)
    assert study["q"] == study["w_ref"].size
    assert np.isfinite(study["f_ref"])
    for N in (3, 4):
        for lvl in study["levels"]:
            ind = study["indicators_by_N"][N][lvl]
            assert ind.shape == (4,) and ind.dtype == bool
    agg = coverage_from_indicators(study["indicators_by_N"][4], deltas=(0.05,))
    for a in agg.values():
        assert 0.0 <= a["coverage"] <= 1.0
        assert a["lower_bounds"][0.05] <= a["coverage"] + 1e-12


def test_coverage_study_threaded_matches_sequential():
    def run(workers):
        solve = _double_integrator_solve()
        sampler = ensemblecontrol.UniformSampler(0.0, 1.0, method="mc", seed=0)
        return coverage_study(sampler, solve, sample_sizes=(3, 4), R=4, n_ref=6,
                              workers=workers)

    seq = run(1)
    par = run(2)                             # explicit 2 -> real outer threads
    assert seq["f_ref"] == par["f_ref"]
    for N in (3, 4):
        for lvl in seq["levels"]:
            assert np.array_equal(seq["indicators_by_N"][N][lvl],
                                  par["indicators_by_N"][N][lvl])


def _subsampling_ci_of(saa, w_opt, f_opt):
    # Mirrors the demo's subsampling ci_of: a fresh rng per call keeps the m
    # subsample index sets deterministic and thread-safe; workers=1 -> serial.
    rng = np.random.default_rng(0)
    rec = ensemblecontrol.subsampling_confidence_interval(
        saa, f_opt, b=3, m=4, rng=rng, w_opt=w_opt,
        levels=(0.90, 0.95, 0.99), workers=1)
    return rec["ci"]


def test_coverage_study_subsampling_ci_of_endtoend():
    def run(workers):
        solve = _double_integrator_solve()
        sampler = ensemblecontrol.UniformSampler(0.0, 1.0, method="mc", seed=1)
        return coverage_study(sampler, solve, sample_sizes=(6,), R=3, n_ref=8,
                              ci_of=_subsampling_ci_of, workers=workers)

    study = run(1)
    for lvl in study["levels"]:
        ind = study["indicators_by_N"][6][lvl]
        assert ind.shape == (3,) and ind.dtype == bool
    agg = coverage_from_indicators(study["indicators_by_N"][6], deltas=(0.05,))
    for a in agg.values():
        assert 0.0 <= a["coverage"] <= 1.0
    # the subsampling ci_of (fresh rng per call) is thread-safe & reproducible
    par = run(2)
    for lvl in study["levels"]:
        assert np.array_equal(study["indicators_by_N"][6][lvl],
                              par["indicators_by_N"][6][lvl])
