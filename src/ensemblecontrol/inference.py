"""Statistical inference for the SAA optimal value.

Implements the two confidence-interval algorithms from "Statistical Inference
for Optimal Values in Scenario-Based Optimal Control under Uncertainty":

  Algorithm 1 (plug-in, unique optimizer):
      sigma_hat_N^2 = (1/N) sum_i (F(x_{u_hat_N}(t_f, xi_i)) - J_hat_N*)^2,
      CI = [ J_hat_N* +/- z_{1-beta/2} sigma_hat_N / sqrt(N) ].

  Algorithm 2 (subsampling, nonunique optimizers):
      Delta_r = sqrt(b) (J*_{I_r} - J_hat_N*)  over m subsamples I_r of size b,
      CI = [ J_hat_N* - q(1-beta/2)/sqrt(N),  J_hat_N* - q(beta/2)/sqrt(N) ],
      where q(p) is the lower empirical p-quantile of the Delta_r.

The compute functions save their raw data -- the per-scenario terminal losses
F_i (plug-in) and the subsampling statistics Delta_r -- to JSON; the confidence
intervals are recomputed from that raw data at plot time (see
:mod:`ensemblecontrol.inference_plotting`), so figures can be restyled,
re-labelled, or re-banded without re-solving.

This module is matplotlib-free.
"""

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
from scipy.stats import norm

from .probability_estimator import probability_lower_bound

__all__ = [
    "terminal_losses",
    "plugin_ci_from_losses", "subsampling_ci_from_deltas",
    "plugin_confidence_interval", "subsampling_confidence_interval",
    "save_plugin_run", "load_plugin_run",
    "save_subsampling_run", "load_subsampling_run",
    "clt_statistic", "save_clt_run", "load_clt_run",
    "coverage_from_indicators", "save_coverage_run", "load_coverage_run",
    "coverage_latex_table",
]


def _scalar(x):
    """Normalize a solve() optimal value (length-1 array or float) to a float."""
    return float(np.ravel(x)[0])


# -- outer/inner parallelism budget ------------------------------------------

# Serializes CasADi/solver GRAPH CONSTRUCTION when the many independent solves
# run in threads (the default subsampling resolver builds each subproblem here);
# the expensive solves run OUTSIDE the lock, so they still overlap.
_BUILD_LOCK = threading.Lock()


# Reserve 2 cores for the OS / main thread: the outer solve loop uses at most
# cpu_count - 2 threads (e.g. 8 on a 10-core machine), matching how the harness
# and the CasADi maps leave headroom rather than pinning every core.
def _core_budget():
    return max(1, (os.cpu_count() or 1) - 2)


def _resolve_workers(workers, njobs, inner_work):
    """Outer worker count so outer x inner threads never exceed the core budget.

    The core budget is ``cpu_count - 2`` (:func:`_core_budget`; 8 on a 10-core
    machine) -- two cores are left free. ``inner_work`` is the per-solve ensemble
    size the inner CasADi map would thread over -- ``b`` subsamples for a
    subsampling re-solve, ``N`` scenarios for a CLT replicate. The two parallelism
    levels must never multiply:

      * ``workers=1`` (default) -- sequential outer loop, inner map left
        threaded; a single heavy solve keeps using the cores. Identical to the
        pre-parallel behavior.
      * ``workers="auto"`` -- returns ``1`` (no outer parallelism) exactly when a
        single solve is already heavy enough to saturate the budget
        (``inner_work >= budget``), else spreads up to ``min(njobs, budget)``
        solves across cores (the caller pins the inner map serial).
      * an integer is passed through (clamped to ``>= 1``); pass an explicit count
        to override the budget when you want more or fewer outer threads.
    """
    budget = _core_budget()
    if workers == "auto":
        return 1 if inner_work >= budget else max(1, min(int(njobs), budget))
    return max(1, int(workers))


# -- per-scenario terminal losses --------------------------------------------

def terminal_losses(saa_problem, w_opt):
    """Per-scenario terminal losses F(x_{u_hat}(t_f, xi_i)) at the optimizer.

    Rolls the optimal control through every scenario with
    ``SAAProblem.ensemble_state_trajectory`` (the same per-sample integrator the
    transcription uses) and applies the model's ``final_cost_function`` to each
    sample's terminal state.  Works for both single and multiple shooting via
    ``SAAProblem.control_matrix``.  Returns a length-N array; its mean equals the
    SAA optimal value when the running cost vanishes (alpha = 0) and beta = 0.
    """
    cp = saa_problem.control_problem
    nstates = cp.nstates
    N = saa_problem.nsamples
    u = saa_problem.control_matrix(w_opt)
    XT = saa_problem.ensemble_state_trajectory(u)[:, -1].reshape(N, nstates)
    return np.array([float(cp.final_cost_function(XT[i])) for i in range(N)])


# -- confidence intervals recomputed from raw data (no solving) --------------

def plugin_ci_from_losses(F, f_opt=None, levels=(0.90, 0.95, 0.99)):
    """Algorithm 1 plug-in interval from the per-scenario losses ``F``.

    Centered at the SAA optimal value J_hat_N* (``f_opt``) when given, else at
    ``mean(F)`` -- the two coincide when alpha = 0 and beta = 0.  The variance
    uses the 1/N (population) denominator, exactly as in the algorithm.  Returns
    a dict ``{N, Jhat, sigma, se, levels: {level: {z, halfwidth, lo, hi}}}``.
    """
    F = np.asarray(F, dtype=float)
    N = F.size
    Jhat = _scalar(f_opt) if f_opt is not None else float(F.mean())
    sigma = float(np.sqrt(np.mean((F - Jhat) ** 2)))
    se = sigma / np.sqrt(N)

    out = {"N": int(N), "Jhat": Jhat, "sigma": sigma, "se": se, "levels": {}}
    for level in levels:
        z = float(norm.ppf(1.0 - (1.0 - level) / 2.0))
        hw = z * se
        out["levels"][level] = {"z": z, "halfwidth": hw,
                                "lo": Jhat - hw, "hi": Jhat + hw}
    return out


def plugin_oos_ci_from_losses(F_oos, f_opt, N, levels=(0.90, 0.95, 0.99)):
    """Plug-in interval with the standard deviation estimated OUT-OF-SAMPLE.

    ``F_oos`` are the terminal losses at the *fixed* optimizer u_hat_N evaluated
    on a fresh, independent sample of size M (M = ``len(F_oos)``).  Unlike the
    in-sample estimator, the variance is centered at the out-of-sample mean (not
    J_hat_N*), giving an unbiased estimate of sigma^2 = Var(F(x_{u_hat_N}(t_f,
    xi))).  The interval is still centered at J_hat_N* (``f_opt``) with standard
    error sigma_oos / sqrt(N), where N is the *training* sample size (not M).
    Returns a dict ``{N, M, Jhat, sigma, se, levels: {level: {z, halfwidth, lo,
    hi}}}``.
    """
    F = np.asarray(F_oos, dtype=float)
    M = F.size
    Jhat = _scalar(f_opt)
    sigma = float(np.sqrt(np.mean((F - F.mean()) ** 2)))   # variance on fresh xi
    se = sigma / np.sqrt(N)

    out = {"N": int(N), "M": int(M), "Jhat": Jhat, "sigma": sigma, "se": se,
           "levels": {}}
    for level in levels:
        z = float(norm.ppf(1.0 - (1.0 - level) / 2.0))
        hw = z * se
        out["levels"][level] = {"z": z, "halfwidth": hw,
                                "lo": Jhat - hw, "hi": Jhat + hw}
    return out


def subsampling_ci_from_deltas(deltas, f_opt, N, levels=(0.90, 0.95, 0.99)):
    """Algorithm 2 subsampling interval from the statistics ``deltas`` (Delta_r).

    ``quantile(p)`` is the lower empirical p-quantile of the sorted Delta_r --
    the value of rank ceil(m p), 1-based.  The interval maps the quantiles back
    through -quantile/sqrt(N) about J_hat_N* (``f_opt``).  Returns a dict
    ``{N, Jhat, m, levels: {level: {quantile_lo, quantile_hi, lo, hi}}}``.
    """
    d = np.sort(np.asarray(deltas, dtype=float))
    m = d.size
    Jhat = _scalar(f_opt)

    def quantile(p):
        rank = int(np.ceil(m * p))
        rank = min(max(rank, 1), m)   # clamp the 1-based rank to [1, m]
        return float(d[rank - 1])

    out = {"N": int(N), "Jhat": Jhat, "m": int(m), "levels": {}}
    for level in levels:
        beta = 1.0 - level
        quantile_hi = quantile(1.0 - beta / 2.0)
        quantile_lo = quantile(beta / 2.0)
        out["levels"][level] = {"quantile_lo": quantile_lo,
                                "quantile_hi": quantile_hi,
                                "lo": Jhat - quantile_hi / np.sqrt(N),
                                "hi": Jhat - quantile_lo / np.sqrt(N)}
    return out


# -- high-level compute (these solve) ----------------------------------------

def plugin_confidence_interval(saa_problem, w_opt, f_opt=None,
                               levels=(0.90, 0.95, 0.99)):
    """Plug-in confidence interval (Algorithm 1) for a solved SAA problem.

    Evaluates the per-scenario terminal losses at the optimizer and forms the
    plug-in interval.  Pass the SAA optimal value as ``f_opt`` so the interval is
    centered at J_hat_N* (faithful to the algorithm; for alpha = 0 it equals
    ``mean(F)``).  Returns a record ``{N, f_opt, F, levels, ci}`` where ``F`` is
    the raw per-scenario loss vector -- the data that gets persisted.
    """
    F = terminal_losses(saa_problem, w_opt)
    Jhat = _scalar(f_opt) if f_opt is not None else float(F.mean())
    ci = plugin_ci_from_losses(F, Jhat, levels)
    return {"N": int(F.size), "f_opt": Jhat, "F": F,
            "q": int(saa_problem.control_problem.nintervals),
            "levels": list(levels), "ci": ci}


def plugin_oos_confidence_interval(saa_problem, w_opt, oos_saa_problem,
                                   f_opt=None, levels=(0.90, 0.95, 0.99)):
    """Plug-in CI (Algorithm 1) with the standard deviation estimated OUT-OF-SAMPLE.

    Rolls the *fixed* optimizer ``w_opt`` (u_hat_N) through the fresh, independent
    scenarios of ``oos_saa_problem`` (size M -- draw it from an independent RNG
    stream, e.g. a spawned reference stream) and estimates sigma from those
    out-of-sample losses instead of the training ones.  ``saa_problem`` is the
    training problem; its size N sets the standard error sigma_oos/sqrt(N).  Pass
    the SAA optimal value ``f_opt`` for the interval center J_hat_N*.  Returns a
    record ``{N, M, f_opt, F, variance, levels, ci}`` where ``F`` is the
    out-of-sample loss vector -- the data that gets persisted.
    """
    F_oos = terminal_losses(oos_saa_problem, w_opt)
    if f_opt is not None:
        Jhat = _scalar(f_opt)
    else:
        Jhat = float(terminal_losses(saa_problem, w_opt).mean())
    N = saa_problem.nsamples
    ci = plugin_oos_ci_from_losses(F_oos, Jhat, N, levels)
    return {"N": int(N), "M": int(F_oos.size), "f_opt": Jhat, "F": F_oos,
            "q": int(saa_problem.control_problem.nintervals),
            "variance": "out-of-sample", "levels": list(levels), "ci": ci}


def _make_progress(progress, N, b, m):
    """Return a callback ``report(done)`` for the subsampling loop.

    ``progress`` is False/None (silent), True (a single-line ``done/m`` counter
    rewritten in place on stderr), or a callable ``report(done, total)`` the
    caller supplies for custom output.
    """
    if not progress:
        return lambda done: None
    if callable(progress):
        return lambda done: progress(done, m)

    def report(done):
        print("\rsubsampling N={} b={}: {}/{} ({:.0%})".format(
            N, b, done, m, done / m),
            end="\n" if done == m else "", file=sys.stderr, flush=True)

    return report


def subsampling_confidence_interval(saa_problem, f_opt, b, m, rng, w_opt=None,
                                    levels=(0.90, 0.95, 0.99), resolve=None,
                                    scipy_tol=1e-5, verbose=False, progress=False,
                                    workers=1):
    """Subsampling confidence interval (Algorithm 2).

    Draws ``m`` subsamples of size ``b`` (uniformly, without replacement) from
    the ``N`` scenarios, re-solves the SAA on each, and forms
    ``Delta_r = sqrt(b) (J*_{I_r} - J_hat_N*)``.  It does NOT re-solve the
    full-sample (size-N) problem: pass its already-computed optimal value
    ``f_opt`` and optimizer ``w_opt`` (e.g. reuse the plug-in solve).

    ``resolve(indices) -> optimal value`` maps a subset of scenario indices to
    the SAA optimal value on that subset.  The default re-solves with scipy
    L-BFGS-B (``ScipyBoxSolver``) warm-started from the full-sample control
    (controls only, never states, via ``SAAProblem.initial_from_controls``); it
    requires ``w_opt`` and a single-shooting, box-only problem
    (``MultipleShooting=False``, ``beta=0``, ``tv_rho=0``).  For other settings
    pass a custom ``resolve``, e.g. Ipopt::

        controls = saa_problem.control_matrix(w_opt)
        def resolve(idx):
            sub = saa_problem.subproblem(idx)
            sub.initial_decisions = sub.initial_from_controls(controls)
            return sub.solve()[1]

    ``progress`` prints a single-line ``r/m`` counter to stderr as it runs
    (default off, preserving silence); pass a callable ``progress(done, total)``
    for custom output.

    ``workers`` controls the parallelism over the ``m`` independent re-solves
    (see :func:`_resolve_workers`): ``1`` (default) is the sequential loop with
    the inner per-sample CasADi map left threaded -- byte-for-byte the old
    behavior; an ``int > 1`` or ``"auto"`` runs the solves in a thread pool with
    the inner map pinned serial (so the two levels never oversubscribe the
    cores). The ``m`` subsample index sets are drawn up front, so ``deltas`` is
    identical for any ``workers``. When threading a *custom* ``resolve`` that
    builds CasADi objects, serialize its construction with the module
    :data:`_BUILD_LOCK` (the default resolver does) or run it at ``workers=1``.

    Returns a record ``{N, f_opt, b, m, deltas, levels, ci}`` where ``deltas`` is
    the raw statistic vector -- the data that gets persisted.
    """
    N = saa_problem.nsamples
    if not (0 < b < N):
        raise ValueError("subsample size b must satisfy 0 < b < N "
                         "(got b=%d, N=%d)" % (b, N))
    Jhat = _scalar(f_opt)
    workers = _resolve_workers(workers, m, inner_work=b)
    inner_serial = workers > 1   # pin the inner map serial when threading solves

    controls_full = None
    if w_opt is not None:
        controls_full = saa_problem.control_matrix(w_opt)

    if resolve is None:
        if controls_full is None:
            raise ValueError("the default subsampling resolver needs w_opt (the "
                             "full-sample optimizer) to warm-start; pass w_opt "
                             "or a custom resolve=.")
        if saa_problem.MultipleShooting or saa_problem.constraints.numel() > 0:
            raise ValueError("the default subsampling resolver requires a "
                             "single-shooting, box-only problem "
                             "(MultipleShooting=False, beta=0, tv_rho=0); pass a "
                             "custom resolve= otherwise.")

        def resolve(indices):
            from .scipy_box import ScipyBoxSolver
            # Build under the lock (CasADi graph construction is the thread-unsafe
            # part); solve outside it so the heavy solves actually overlap.
            with _BUILD_LOCK:
                sub = saa_problem.subproblem(
                    indices,
                    parallelization="serial" if inner_serial else None,
                    n_threads=1 if inner_serial else None)
                w0 = sub.initial_from_controls(controls_full)
                solver = ScipyBoxSolver(sub, method="L-BFGS-B", tol=scipy_tol,
                                        verbose=verbose)
            return solver.solve(w0=w0)[1]

    # Draw all m subsample index sets up front: the RNG draws do not depend on
    # the solve results, so the draw order -- hence the deltas -- is identical
    # for any `workers`, keeping the interval reproducible under parallelism.
    index_sets = [rng.choice(N, size=b, replace=False) for _ in range(m)]

    def _delta(indices):
        return np.sqrt(b) * (_scalar(resolve(indices)) - Jhat)

    deltas = np.empty(m)
    report = _make_progress(progress, N, b, m)
    if workers == 1:
        for r, indices in enumerate(index_sets):
            deltas[r] = _delta(indices)
            report(r + 1)
    else:
        # ThreadPoolExecutor.map preserves input order, so deltas stays aligned
        # to index_sets (and thus to the sequential result).
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for r, d in enumerate(ex.map(_delta, index_sets)):
                deltas[r] = d
                report(r + 1)

    ci = subsampling_ci_from_deltas(deltas, Jhat, N, levels)
    return {"N": int(N), "f_opt": Jhat, "b": int(b), "m": int(m),
            "deltas": deltas, "q": int(saa_problem.control_problem.nintervals),
            "levels": list(levels), "ci": ci}


# -- raw-data persistence (JSON primary; human-readable txt/csv secondary) ----

def _created(meta):
    out = dict(meta) if meta else {}
    out.setdefault("created", datetime.now().isoformat(timespec="seconds"))
    return out


def _write_json(path, data):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)


def save_plugin_run(records, path, meta=None, write_tables=True, r=None):
    """Persist plug-in raw data (per-N f_opt and loss vectors F) to ``path`` as
    JSON.

    ``records`` is a single record from :func:`plugin_confidence_interval` or a
    list of them (one per sample size).  Only the raw data and the confidence
    ``levels`` are stored; the intervals are recomputed from ``F`` at plot time.
    ``r`` (the scenario radius) is stored once at run level so plots can list it
    alongside ``q``.  With ``write_tables`` also writes human-readable
    ``.txt``/``.csv`` next to the JSON.  Returns ``path``.
    """
    if isinstance(records, dict):
        records = [records]
    levels = list(records[0]["levels"]) if records else []
    # q (control-mesh size = number of intervals) is constant across the
    # sample-size sweep; store it once at run level so plots can list it.
    q = records[0].get("q") if records else None
    # in-sample or out-of-sample standard-deviation estimate (records from
    # plugin_oos_confidence_interval carry variance="out-of-sample").
    variance = records[0].get("variance", "in-sample") if records else "in-sample"
    results = []
    for rec in records:                    # not `r`: `r` is the run-level radius arg
        res = {"N": int(rec["N"]), "f_opt": float(rec["f_opt"]),
               "F": np.asarray(rec["F"], dtype=float).tolist()}
        if "M" in rec:                     # out-of-sample size (F is the OOS losses)
            res["M"] = int(rec["M"])
        results.append(res)
    data = {
        "algorithm": "plugin",
        "variance": variance,
        "levels": levels,
        "q": q,
        "r": r,
        "results": results,
        "meta": _created(meta),
    }
    _write_json(path, data)
    if write_tables:
        _write_plugin_tables(path, data)
    return path


def load_plugin_run(path):
    """Load a plug-in run saved by :func:`save_plugin_run`.

    Returns ``{algorithm, levels, results: [{N, f_opt, F(ndarray)}...], meta}``.
    """
    with open(path) as fh:
        data = json.load(fh)
    for r in data["results"]:
        r["F"] = np.asarray(r["F"], dtype=float)
    return data


def save_subsampling_run(records, path, meta=None, write_tables=True, r=None):
    """Persist subsampling raw data (the Delta_r statistics) to ``path`` as JSON.

    ``records`` is a single record from
    :func:`subsampling_confidence_interval` or a list of them (one per sample
    size N -- the subsampling analogue of the plug-in sweep).  Only the raw
    ``deltas`` and the run parameters are stored; the interval is recomputed at
    plot time.  ``r`` (the scenario radius) is stored once at run level so plots
    can list it alongside ``q``.  Returns ``path``.
    """
    if isinstance(records, dict):
        records = [records]
    levels = list(records[0]["levels"]) if records else []
    # q (control-mesh size) is constant across the sweep; store it once.
    q = records[0].get("q") if records else None
    data = {
        "algorithm": "subsampling",
        "levels": levels,
        "q": q,
        "r": r,
        "results": [{"N": int(rec["N"]), "f_opt": float(rec["f_opt"]),
                     "b": int(rec["b"]), "m": int(rec["m"]),
                     "deltas": np.asarray(rec["deltas"], dtype=float).tolist()}
                    for rec in records],
        "meta": _created(meta),
    }
    _write_json(path, data)
    if write_tables:
        _write_subsampling_tables(path, data)
    return path


def load_subsampling_run(path):
    """Load a subsampling run saved by :func:`save_subsampling_run`.

    Returns ``{algorithm, levels, q, results: [{N, f_opt, b, m, deltas(ndarray)}
    ...], meta}``.  A legacy single-N file (flat ``N/f_opt/b/m/deltas``) is
    upgraded to the one-element ``results`` form on load.
    """
    with open(path) as fh:
        data = json.load(fh)
    if "results" not in data:   # legacy single-N schema
        data = {"algorithm": "subsampling", "levels": data.get("levels", []),
                "q": data.get("q"),
                "results": [{"N": data["N"], "f_opt": data["f_opt"],
                             "b": data["b"], "m": data["m"],
                             "deltas": data["deltas"]}],
                "meta": data.get("meta", {})}
    for r in data["results"]:
        r["deltas"] = np.asarray(r["deltas"], dtype=float)
    return data


def _write_plugin_tables(json_path, data):
    base = os.path.splitext(json_path)[0]
    levels = data["levels"]
    oos = data.get("variance") == "out-of-sample"
    if oos:
        cis = [plugin_oos_ci_from_losses(np.asarray(r["F"], float), r["f_opt"],
                                         r["N"], levels) for r in data["results"]]
        header = ["Plug-in confidence intervals for the SAA optimal value "
                  "(Algorithm 1)",
                  "standard deviation estimated OUT-OF-SAMPLE:",
                  "sigma_oos^2 = (1/M) sum_j (F'_j - mean(F'))^2 at u_hat_N on "
                  "M fresh scenarios", ""]
    else:
        cis = [plugin_ci_from_losses(np.asarray(r["F"], float), r["f_opt"], levels)
               for r in data["results"]]
        header = ["Plug-in confidence intervals for the SAA optimal value "
                  "(Algorithm 1)",
                  "sigma_hat_N^2 = (1/N) sum_i (F_i - J_hat_N*)^2", ""]

    lines = list(header)
    for ci in cis:
        lines.append("N = {}{}".format(ci["N"], "  (M = {})".format(ci["M"])
                                       if "M" in ci else ""))
        lines.append("  J_hat_N*            = {: .8e}".format(ci["Jhat"]))
        lines.append("  sigma{}         = {: .8e}".format(
            "_oos " if oos else "_hat_N", ci["sigma"]))
        lines.append("  sigma/sqrt(N)       = {: .8e}".format(ci["se"]))
        for level in levels:
            bnd = ci["levels"][level]
            lines.append("  {:.0%} CI: z = {:.6f}  half-width = {: .6e}  "
                         "[{: .8e}, {: .8e}]".format(level, bnd["z"],
                         bnd["halfwidth"], bnd["lo"], bnd["hi"]))
        lines.append("")
    with open(base + ".txt", "w") as fh:
        fh.write("\n".join(lines))

    for r in data["results"]:
        np.savetxt("{}_losses_N{}.csv".format(base, r["N"]),
                   np.asarray(r["F"], dtype=float), delimiter=",",
                   header="F", comments="")

    rows = []
    for ci in cis:
        for level in levels:
            bnd = ci["levels"][level]
            rows.append([ci["N"], level, bnd["z"], ci["Jhat"], ci["sigma"],
                         ci["se"], bnd["halfwidth"], bnd["lo"], bnd["hi"]])
    if rows:
        np.savetxt(base + "_ci.csv", np.asarray(rows, dtype=float),
                   delimiter=",",
                   header="N,level,z,Jhat,sigma,se,halfwidth,lo,hi", comments="")


def _write_subsampling_tables(json_path, data):
    base = os.path.splitext(json_path)[0]
    levels = data["levels"]
    lines = ["Subsampling confidence intervals for the SAA optimal value "
             "(Algorithm 2)",
             "Delta_r = sqrt(b) (J*_I_r - J_hat_N*)", ""]
    for r in data["results"]:
        deltas = np.asarray(r["deltas"], float)
        ci = subsampling_ci_from_deltas(deltas, r["f_opt"], r["N"], levels)
        lines.append("N = {}  (b = {}, m = {})".format(r["N"], r["b"], r["m"]))
        lines.append("  J_hat_N* = {: .8e}".format(r["f_opt"]))
        for level in levels:
            bnd = ci["levels"][level]
            lines.append("  {:.0%} CI: quantile_lo = {: .6e}  "
                         "quantile_hi = {: .6e}  [{: .8e}, {: .8e}]".format(
                             level, bnd["quantile_lo"], bnd["quantile_hi"],
                             bnd["lo"], bnd["hi"]))
        lines.append("")
        np.savetxt("{}_deltas_N{}.csv".format(base, r["N"]), deltas,
                   delimiter=",", header="delta", comments="")
    with open(base + ".txt", "w") as fh:
        fh.write("\n".join(lines))


# -- central-limit-theorem illustration --------------------------------------

def clt_statistic(values, N, f_ref):
    """The scaled optimal-value error sqrt(N)*(J_hat_N* - J_hat_ref*).

    ``values`` are the R replicate SAA optimal values at sample size ``N``, and
    ``f_ref`` is the reference SAA value J_hat_ref* (on an independent sample of
    size N_ref) proxying the population optimal value J*.  The statistical limit
    theorem predicts this statistic converges to a centered Gaussian as N grows
    (under a unique population optimizer).
    """
    return np.sqrt(N) * (np.asarray(values, dtype=float) - float(f_ref))


def save_clt_run(values_by_N, path, N_ref, f_ref, q=None, meta=None,
                 write_tables=True, r=None):
    """Persist a CLT replication study to ``path`` as JSON.

    ``values_by_N`` is ``{N: array of R replicate SAA optimal values}``; ``N_ref``
    and ``f_ref`` are the reference sample size and its SAA value J_hat_ref*.  The
    RAW (unscaled) replicate values are stored -- the statistic
    sqrt(N)*(value - f_ref) is recomputed at plot time (see plot_clt), so the
    histograms can be re-drawn without re-running the study.  ``q`` is the control
    mesh size and ``r`` the scenario radius (both listed in the plot legends).
    With ``write_tables`` also writes the per-N statistics CSV and a summary txt.
    Returns ``path``.
    """
    data = {
        "algorithm": "clt",
        "N_ref": int(N_ref),
        "f_ref": float(f_ref),
        "q": q,
        "r": r,
        "results": [{"N": int(N),
                     "values": np.asarray(values_by_N[N], dtype=float).tolist()}
                    for N in sorted(values_by_N)],
        "meta": _created(meta),
    }
    _write_json(path, data)
    if write_tables:
        _write_clt_tables(path, data)
    return path


def load_clt_run(path):
    """Load a CLT run saved by :func:`save_clt_run`.

    Returns ``{algorithm, N_ref, f_ref, q, results: [{N, values(ndarray)}...],
    meta}``.
    """
    with open(path) as fh:
        data = json.load(fh)
    for r in data["results"]:
        r["values"] = np.asarray(r["values"], dtype=float)
    return data


def _write_clt_tables(json_path, data):
    base = os.path.splitext(json_path)[0]
    N_ref = data["N_ref"]
    f_ref = data["f_ref"]
    results = data["results"]
    stats = [clt_statistic(r["values"], r["N"], f_ref) for r in results]

    # one column per N of the scaled statistic sqrt(N)*(J_hat_N* - J_hat_ref*)
    np.savetxt(base + "_statistics.csv", np.column_stack(stats), delimiter=",",
               header=",".join("sqrtN_dJ_N{}".format(r["N"]) for r in results),
               comments="")

    lines = ["Statistical limit theorem illustration",
             "statistic = sqrt(N) * (J_hat_N* - J_hat_ref*)",
             "N_ref = {}".format(N_ref),
             "J_hat_ref* = {: .8e}".format(f_ref), ""]
    for r, s in zip(results, stats):
        lines.append("N = {}  (R = {} replicates)".format(r["N"], s.size))
        lines.append("  mean = {: .6e}   std = {: .6e}".format(
            float(s.mean()), float(s.std())))
        lines.append("")
    with open(base + ".txt", "w") as fh:
        fh.write("\n".join(lines))


# -- coverage test (Monte-Carlo validation of the confidence intervals) -------

def coverage_from_indicators(indicators, deltas=(0.05,)):
    """Aggregate one sample size's coverage indicators into coverage bounds.

    ``indicators`` is ``{level: array of R booleans}`` (did the level-``level`` CI
    of replication r cover the reference value?).  For each level returns the
    covered count L, the sample size R, the empirical coverage L/R, and the
    (1-delta) lower confidence bounds :func:`probability_lower_bound` ``(R, L,
    delta)`` for each delta -- a rigorous lower bound on the interval's true
    coverage probability (``delta`` is the failure probability of *that* bound,
    not the CI's confidence level).  Returns ``{level: {L, R, coverage,
    lower_bounds: {delta: p_lower}}}``.  Pure; does no solving.
    """
    out = {}
    for level, ind in indicators.items():
        ind = np.asarray(ind, dtype=bool)
        R = int(ind.size)
        L = int(ind.sum())
        out[level] = {
            "L": L, "R": R,
            "coverage": (L / R) if R else float("nan"),
            "lower_bounds": {d: probability_lower_bound(R, L, d) for d in deltas},
        }
    return out


def save_coverage_run(study, path, meta=None, write_tables=True, r=None):
    """Persist a coverage study (from :func:`coverage_study`) to ``path`` as JSON.

    Only the RAW per-N per-level coverage indicators and the run parameters
    (reference value ``f_ref``, ``n_ref``, ``levels``, ``R``, mesh size ``q``,
    scenario radius ``r``) are stored; the empirical coverage and the lower
    confidence bounds are recomputed from the indicators at report time (see
    :func:`coverage_from_indicators` / :func:`coverage_latex_table`), so the table
    can be re-derived without re-running the ~R*len(N) solves.  Indicators are
    stored as 0/1 columns aligned with ``levels`` (avoiding float JSON keys).
    With ``write_tables`` also writes a human-readable ``.txt`` summary.  Returns
    ``path``.
    """
    levels = list(study["levels"])
    ind_by_N = study["indicators_by_N"]
    results = []
    for N in sorted(ind_by_N):
        per = ind_by_N[N]
        results.append({
            "N": int(N),
            "indicators": [np.asarray(per[level], dtype=bool).astype(int).tolist()
                           for level in levels],
        })
    data = {
        "algorithm": "coverage",
        "levels": levels,
        "n_ref": int(study["n_ref"]),
        "f_ref": float(study["f_ref"]),
        "q": study.get("q"),
        "r": r,
        "R": int(study["R"]),
        "results": results,
        "meta": _created(meta),
    }
    _write_json(path, data)
    if write_tables:
        _write_coverage_tables(path, data)
    return path


def load_coverage_run(path):
    """Load a coverage run saved by :func:`save_coverage_run`.

    Returns ``{algorithm, levels, n_ref, f_ref, q, r, R, results: [{N,
    indicators: {level: ndarray of bool}}...], meta}`` -- the indicator columns
    are zipped back onto ``levels`` as a ``{level: array}`` dict per N.
    """
    with open(path) as fh:
        data = json.load(fh)
    for res in data["results"]:
        res["indicators"] = {level: np.asarray(col, dtype=bool)
                             for level, col in zip(data["levels"],
                                                   res["indicators"])}
    return data


def _coverage_rows(run_or_path):
    """Normalize a coverage study/run/path into (levels, [(N, {level: array})])."""
    data = load_coverage_run(run_or_path) if isinstance(run_or_path, str) \
        else run_or_path
    levels = list(data["levels"])
    if "indicators_by_N" in data:      # in-memory study from coverage_study
        rows = [(int(N), data["indicators_by_N"][N])
                for N in data["sample_sizes"]]
    else:                              # saved/loaded run (results list)
        rows = [(int(res["N"]),
                 res["indicators"] if isinstance(res["indicators"], dict)
                 else {lvl: np.asarray(col, dtype=bool)
                       for lvl, col in zip(levels, res["indicators"])})
                for res in data["results"]]
    return levels, rows


def _delta_tex(d):
    """LaTeX subscript for a delta value: a power of ten as 10^{k}, else decimal."""
    e = round(float(np.log10(d)))
    if np.isclose(d, 10.0 ** e):
        return "10^{{{}}}".format(int(e))
    return "{:g}".format(d)


def coverage_latex_table(run_or_path, deltas=(0.05,), levels=None,
                         caption=None, label=None):
    """LaTeX (booktabs) table of the estimated coverage probabilities.

    Rows are the sample sizes N; columns are grouped by nominal CI level, each
    group showing the empirical coverage L/R and the (1-delta) lower confidence
    bound(s) :func:`probability_lower_bound` ``(R, L, delta)``.  ``run_or_path`` is
    a coverage study (from :func:`coverage_study`), a loaded/saved run dict, or a
    JSON path from :func:`save_coverage_run`.  ``levels`` selects/orders a subset
    of the stored levels (default: all).  Requires the ``booktabs`` package.
    Returns the table as a string.
    """
    all_levels, rows = _coverage_rows(run_or_path)
    levels = list(levels) if levels is not None else all_levels
    deltas = list(deltas)
    per_level = 1 + len(deltas)                     # L/R plus one bound per delta

    group = [""]
    cmid, start = [], 2
    for level in levels:
        group.append("\\multicolumn{{{}}}{{c}}{{$1-\\alpha = {:.2f}$}}".format(
            per_level, level))
        end = start + per_level - 1
        cmid.append("\\cmidrule(lr){{{}-{}}}".format(start, end))
        start = end + 1

    sub = ["$N$"]
    for _ in levels:
        sub.append("$L/R$")
        sub.extend("$\\underline{{p}}_{{{}}}$".format(_delta_tex(d)) for d in deltas)

    body = []
    for N, indicators in rows:
        agg = coverage_from_indicators(indicators, deltas)
        cells = ["{}".format(N)]
        for level in levels:
            a = agg[level]
            cells.append("{:.3f}".format(a["coverage"]))
            cells.extend("{:.3f}".format(a["lower_bounds"][d]) for d in deltas)
        body.append(" & ".join(cells) + r" \\")

    colspec = "r" + "c" * (len(levels) * per_level)
    lines = [r"\begin{table}[t]", r"  \centering"]
    if caption:
        lines.append("  \\caption{{{}}}".format(caption))
    if label:
        lines.append("  \\label{{{}}}".format(label))
    lines.append("  \\begin{{tabular}}{{{}}}".format(colspec))
    lines.append("    \\toprule")
    lines.append("    " + " & ".join(group) + r" \\")
    lines.append("    " + "".join(cmid))
    lines.append("    " + " & ".join(sub) + r" \\")
    lines.append("    \\midrule")
    lines.extend("    " + row for row in body)
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def _write_coverage_tables(json_path, data):
    base = os.path.splitext(json_path)[0]
    levels = data["levels"]
    # Name the CI in the header from meta["ci"] when the caller recorded it.
    ci = (data.get("meta") or {}).get("ci", "")
    label = {"plugin": "plug-in", "subsampling": "subsampling"}.get(ci, ci or "SAA")
    lines = ["Monte-Carlo coverage test of the %s confidence interval" % label,
             "coverage = (# CIs covering J_hat_ref*) / R",
             "p_lower  = (1-delta) lower bound on the true coverage (delta = 0.05)",
             "J_hat_ref* = {: .8e}  (N_ref = {})".format(data["f_ref"],
                                                         data["n_ref"]),
             "R = {} replications per N".format(data["R"]), ""]
    _, rows = _coverage_rows(data)
    for N, indicators in rows:
        agg = coverage_from_indicators(indicators, deltas=(0.05,))
        lines.append("N = {}".format(N))
        for level in levels:
            a = agg[level]
            lines.append("  level {:.2f}:  coverage = {}/{} = {:.4f}   "
                         "p_lower = {:.4f}".format(level, a["L"], a["R"],
                                                   a["coverage"],
                                                   a["lower_bounds"][0.05]))
        lines.append("")
    with open(base + ".txt", "w") as fh:
        fh.write("\n".join(lines))
