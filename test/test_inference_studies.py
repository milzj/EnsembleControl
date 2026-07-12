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


def test_subsampling_sweep_default_m_is_5N_per_size():
    # New default: m_N = 5N per sample size (was constant m = 5*max(N)).
    model = DoubleIntegrator()
    solve = ensemblecontrol.make_scipy_solve(model, tol=1e-6)
    solves = ensemblecontrol.solve_saa_prefixes(solve, [[0]] * 16, (8, 16))
    recs = ensemblecontrol.subsampling_sweep(solves, (8, 16), seed=3, workers=1)
    assert [r["m"] for r in recs] == [40, 80]          # 5*8, 5*16 (not 80, 80)
    # a callable m schedule is honored per N; a scalar stays constant across N
    cb = ensemblecontrol.subsampling_sweep(solves, (8, 16), m=lambda N: N, seed=3)
    assert [r["m"] for r in cb] == [8, 16]
    const = ensemblecontrol.subsampling_sweep(solves, (8, 16), m=7, seed=3)
    assert [r["m"] for r in const] == [7, 7]


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


# -- monotonicity diagnostic --------------------------------------------------

def test_monotonicity_check():
    R = 400
    alt = np.empty(R)                 # zero-mean, unit-std pattern (200x +1, 200x -1)
    alt[0::2], alt[1::2] = 1.0, -1.0
    base = np.linspace(-1.0, 1.0, R)  # arbitrary shared (nested/CRN) component
    # Paired differences are constructed directly so each pair lands robustly in one
    # bucket: (8->16) delta=+0.5 -> OK; (16->32) delta=-0.1 with se~0.10 -> z~-1
    # (compatible with MC noise); (32->64) delta=-0.5 with se~5e-5 -> z~-1e4 (violation).
    v8 = base - 0.5
    v16 = base
    v32 = base + (-0.1 + 2.0 * alt)
    v64 = v32 + (-0.5 + 0.001 * alt)
    values = {8: v8, 16: v16, 32: v32, 64: v64}

    rows = ensemblecontrol.monotonicity_check(values, nested=True)
    assert [(r["N1"], r["N2"]) for r in rows] == [(8, 16), (16, 32), (32, 64)]
    by_pair = {(r["N1"], r["N2"]): r for r in rows}
    assert by_pair[(8, 16)]["status"] == "OK"                 # delta > 0
    assert by_pair[(16, 32)]["status"] == "compatible with MC noise"
    assert by_pair[(32, 64)]["status"] == "potential issue"   # z < -2

    # nested se is exactly std(v2 - v1, ddof=1)/sqrt(R)
    r = by_pair[(16, 32)]
    D = v32 - v16
    assert np.isclose(r["se"], D.std(ddof=1) / np.sqrt(R))
    assert np.isclose(r["z"], r["delta"] / r["se"])

    # non-nested se is the independent combination sqrt(se1^2 + se2^2), and (because the
    # columns are positively correlated here) differs from the paired se.
    unp = {(x["N1"], x["N2"]): x
           for x in ensemblecontrol.monotonicity_check(values, nested=False)}
    se1 = v16.std(ddof=1) / np.sqrt(R)
    se2 = v32.std(ddof=1) / np.sqrt(R)
    assert np.isclose(unp[(16, 32)]["se"], np.hypot(se1, se2))
    assert not np.isclose(unp[(16, 32)]["se"], r["se"])

    # nested=True with unequal R is rejected
    try:
        ensemblecontrol.monotonicity_check({8: np.zeros(5), 16: np.zeros(6)},
                                           nested=True)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unequal R under nested=True")

    # formatter renders one line per pair with the status text
    table = ensemblecontrol.format_monotonicity_table(rows)
    assert "status" in table and "potential issue" in table
    assert len(table.splitlines()) == 2 + len(rows)   # header + rule + rows


def test_plot_monotonicity_smoke(tmp_path):
    # renders from a loaded-run-style dict (has "results")
    run = {"q": 5, "r": 0.04,
           "results": [{"N": 8, "values": np.array([-1.2, -1.1, -1.3, -1.15])},
                       {"N": 16, "values": np.array([-1.05, -1.0, -1.1, -1.02])},
                       {"N": 32, "values": np.array([-1.06, -1.01, -1.11, -1.03])}]}
    paths = ensemblecontrol.plot_monotonicity(run, outdir=str(tmp_path),
                                              formats=("png",))
    assert len(paths) == 1 and os.path.isfile(paths[0])
    assert paths[0].endswith("monotonicity.png")
