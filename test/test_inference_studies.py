import matplotlib
matplotlib.use("Agg")  # must precede any pyplot import (ensemblecontrol pulls in pyplot)

import os  # noqa: E402
import numpy as np  # noqa: E402

import ensemblecontrol  # noqa: E402

from .double_integrator import DoubleIntegrator  # noqa: E402


# -- default subsampling schedule --------------------------------------------

def test_default_subsample_schedule():
    assert ensemblecontrol.default_subsample_size(128) == 64        # floor(128^{6/7})
    assert ensemblecontrol.default_subsample_size(32) == \
        int(np.floor(32 ** (6.0 / 7.0) + 1e-9))
    assert ensemblecontrol.default_num_subsamples(128) == 640       # 5 * 128


# -- sample-size sweep helpers ------------------------------------------------

def test_solve_prefixes_and_plugin_sweep():
    model = DoubleIntegrator()
    solve = ensemblecontrol.make_scipy_solve(model, tol=1e-6)
    samples = [[0]] * 8
    solves = ensemblecontrol.solve_saa_prefixes(solve, samples, (4, 8))
    assert set(solves) == {4, 8}
    saa4, w4, f4 = solves[4]
    assert saa4.nsamples == 4 and np.isfinite(f4)

    recs = ensemblecontrol.plugin_sweep(solves, (4, 8))
    assert [r["N"] for r in recs] == [4, 8]
    # the sweep is exactly the per-N plugin_confidence_interval on the same solves
    direct = ensemblecontrol.plugin_confidence_interval(saa4, w4, f_opt=f4)
    assert np.allclose(recs[0]["F"], direct["F"])


def test_subsampling_sweep_matches_direct_call():
    model = DoubleIntegrator()
    solve = ensemblecontrol.make_scipy_solve(model, tol=1e-6)
    solves = ensemblecontrol.solve_saa_prefixes(solve, [[0]] * 10, (10,))
    recs = ensemblecontrol.subsampling_sweep(
        solves, (10,), b_of=lambda N: 4, m=5, seed=2, workers=1)
    assert len(recs) == 1 and (recs[0]["b"], recs[0]["m"]) == (4, 5)

    # subsampling_sweep spawns one index stream per N from `seed`; reproduce it.
    ss = np.random.SeedSequence(2).spawn(1)[0]
    saa, w, f = solves[10]
    direct = ensemblecontrol.subsampling_confidence_interval(
        saa, f, b=4, m=5, rng=np.random.default_rng(ss), w_opt=w)
    assert np.allclose(recs[0]["deltas"], direct["deltas"])


def test_subsampling_sweep_rejects_bad_block_size():
    model = DoubleIntegrator()
    solve = ensemblecontrol.make_scipy_solve(model, tol=1e-6)
    solves = ensemblecontrol.solve_saa_prefixes(solve, [[0]] * 6, (6,))
    try:
        ensemblecontrol.subsampling_sweep(solves, (6,), b_of=lambda N: 6, m=2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for b >= N")


# -- CLT replication study ----------------------------------------------------

def test_clt_replication_study_workers_equivalence():
    # A deterministic fake solve (value = sum of the drawn scenarios) makes the
    # replicate values nonconstant and order-sensitive, isolating the study's
    # threading/ordering from any real solver. No model needed.
    def fake_solve(samples, w0=None, inner_serial=False):
        return None, np.zeros(3), float(np.asarray(samples, dtype=float).sum())

    def fresh():
        return ensemblecontrol.UniformSampler(0.0, 1.0, method="mc", seed=5)

    def study(workers):
        return ensemblecontrol.clt_replication_study(
            fresh(), fake_solve, sample_sizes=(6, 10), R=15, n_ref=8,
            workers=workers)

    a = study(1)
    b = study("auto")   # N=6 < budget -> threaded; N=10 may be serial
    c = study(4)
    assert a["n_ref"] == 8 and a["q"] == 3
    assert a["f_ref"] == b["f_ref"] == c["f_ref"]
    for N in (6, 10):
        assert a["values_by_N"][N].shape == (15,)
        assert a["values_by_N"][N].std() > 0                       # nonconstant
        assert np.array_equal(a["values_by_N"][N], b["values_by_N"][N])
        assert np.array_equal(a["values_by_N"][N], c["values_by_N"][N])


def test_clt_replication_study_with_real_solver_roundtrips(tmp_path):
    # End-to-end wiring: make_scipy_solve + study + save_clt_run (DoubleIntegrator
    # ignores the parameter, so the values are constant -- we only check plumbing).
    model = DoubleIntegrator()
    solve = ensemblecontrol.make_scipy_solve(model, tol=1e-6)
    sampler = ensemblecontrol.UniformSampler(0.0, 1.0, method="mc", seed=0)
    study = ensemblecontrol.clt_replication_study(
        sampler, solve, sample_sizes=(4, 6), R=3, n_ref=8, workers=1)
    assert set(study["values_by_N"]) == {4, 6}
    assert study["w_ref"].size == study["q"]

    path = ensemblecontrol.save_clt_run(
        study["values_by_N"], os.path.join(tmp_path, "clt.json"),
        N_ref=study["n_ref"], f_ref=study["f_ref"], q=study["q"])
    run = ensemblecontrol.load_clt_run(path)
    assert [r["N"] for r in run["results"]] == [4, 6]


def test_clt_replication_study_uses_common_random_numbers():
    # Within each replicate the size-N samples must be the nested PREFIX of the
    # size-max samples (common random numbers across N -- the canonical SAA
    # construction). A recording fake solve captures the samples each solve receives.
    seen = {}

    def rec_solve(samples, w0=None, inner_serial=False):
        s = np.asarray(samples, dtype=float)
        seen.setdefault(s.shape[0], []).append(s.copy())
        return None, np.zeros(3), float(s.sum())

    sampler = ensemblecontrol.UniformSampler(0.0, 1.0, method="mc", seed=7)
    ensemblecontrol.clt_replication_study(
        sampler, rec_solve, sample_sizes=(6, 10), R=5, n_ref=8, workers=1)

    for r in range(5):
        # replicate r's size-6 problem is the first 6 scenarios of its size-10 problem
        assert np.allclose(seen[6][r], seen[10][r][:6])
    # independent replicates draw different sequences
    assert not np.allclose(seen[10][0], seen[10][1])


def test_plot_optimization_bias_smoke(tmp_path):
    # single-panel bias diagnostic (mean E[Jhat_N*] + reference line) from a CLT run dict
    run = {"N_ref": 64, "f_ref": -1.0, "q": 5, "r": 0.04,
           "results": [{"N": 4, "values": np.array([-1.2, -1.1, -1.3])},
                       {"N": 8, "values": np.array([-1.05, -1.0, -1.1])}]}
    paths = ensemblecontrol.plot_optimization_bias(run, outdir=str(tmp_path),
                                                   formats=("png",))
    assert len(paths) == 1 and os.path.isfile(paths[0])
    assert paths[0].endswith("optimization_bias.png")
