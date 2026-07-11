import matplotlib
matplotlib.use("Agg")  # must precede any pyplot import (ensemblecontrol pulls in pyplot)

import os  # noqa: E402
import numpy as np  # noqa: E402

import ensemblecontrol  # noqa: E402


NOMINAL = [4.75, 0.12, -5, 0.1, 21.87, 0.4, 62.5, -7.3]


def _sampler(seed=0):
    return ensemblecontrol.UniformRelativeSampler(NOMINAL, radius=0.04, seed=seed)


def _recording_solve(store=None, q=5):
    """Fake ``solve(samples, w0, inner_serial)`` that does NO optimization: returns
    ``f = -mean(samples)`` and a zero decision vector of size ``q``.  When ``store`` is
    given it records each received sample array keyed by N, so a test can inspect the
    common-random-number nesting.  With ``store=None`` it is thread-safe.
    """
    def solve(samples, w0=None, inner_serial=False):
        s = np.asarray(samples, dtype=float)
        if store is not None:
            store.setdefault(s.shape[0], []).append(s.copy())
        return None, np.zeros(q), float(-s.mean())
    return solve


def test_study_shapes_and_q():
    run = ensemblecontrol.optimal_value_study(
        _sampler(), _recording_solve(q=7), sample_sizes=[8, 16, 32], R=10, workers=1)
    assert set(run["values_by_N"]) == {8, 16, 32}
    for N in (8, 16, 32):
        assert run["values_by_N"][N].shape == (10,)
    assert run["q"] == 7


def test_common_random_numbers_nesting():
    # each replicate's size-N samples must be the nested PREFIX of its size-max samples
    store = {}
    ensemblecontrol.optimal_value_study(
        _sampler(seed=1), _recording_solve(store), sample_sizes=[8, 16, 32], R=6,
        workers=1)
    for r in range(6):
        assert np.allclose(store[8][r], store[16][r][:8])
        assert np.allclose(store[8][r], store[32][r][:8])
        assert np.allclose(store[16][r], store[32][r][:16])
    # different replicates are independent draws
    assert not np.allclose(store[32][0], store[32][1])


def test_determinism_same_seed():
    a = ensemblecontrol.optimal_value_study(
        _sampler(seed=3), _recording_solve(), [8, 16], R=8, workers=1)
    b = ensemblecontrol.optimal_value_study(
        _sampler(seed=3), _recording_solve(), [8, 16], R=8, workers=1)
    for N in (8, 16):
        assert np.array_equal(a["values_by_N"][N], b["values_by_N"][N])


def test_workers_auto_matches_serial():
    serial = ensemblecontrol.optimal_value_study(
        _sampler(seed=2), _recording_solve(), [8, 16], R=12, workers=1)
    auto = ensemblecontrol.optimal_value_study(
        _sampler(seed=2), _recording_solve(), [8, 16], R=12, workers="auto")
    for N in (8, 16):
        assert np.allclose(serial["values_by_N"][N], auto["values_by_N"][N])


def test_mean_value_series():
    vals = {8: np.array([1.0, 3.0, 5.0]), 16: np.array([2.0, 2.0, 2.0])}
    N, mean, se = ensemblecontrol.mean_value_series({"values_by_N": vals})
    assert np.allclose(N, [8, 16])
    assert np.allclose(mean, [3.0, 2.0])
    assert np.isclose(se[0], np.std([1.0, 3.0, 5.0], ddof=1) / np.sqrt(3))
    assert np.isclose(se[1], 0.0)


def test_save_load_roundtrip(tmp_path):
    run = ensemblecontrol.optimal_value_study(
        _sampler(), _recording_solve(), [8, 16], R=5, workers=1)
    path = ensemblecontrol.save_optimal_value_run(
        run, str(tmp_path / "mv.json"), r=0.04, meta={"model": "fake"})
    loaded = ensemblecontrol.load_optimal_value_run(path)
    assert loaded["r"] == 0.04
    assert loaded["q"] == run["q"]
    for rec in loaded["results"]:
        assert np.allclose(rec["values"], run["values_by_N"][rec["N"]])
    # mean_value_series must agree whether fed the study dict or the loaded run
    assert np.allclose(ensemblecontrol.mean_value_series(run)[1],
                       ensemblecontrol.mean_value_series(loaded)[1])


def test_plot_smoke(tmp_path):
    run = ensemblecontrol.optimal_value_study(
        _sampler(), _recording_solve(), [8, 16, 32], R=5, workers=1)
    run["r"] = 0.04
    paths = ensemblecontrol.plot_optimal_value(run, outdir=str(tmp_path),
                                               formats=("png",))
    assert len(paths) == 1 and os.path.isfile(paths[0])
