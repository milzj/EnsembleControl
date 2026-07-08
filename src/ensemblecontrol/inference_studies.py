"""Model-agnostic orchestration for the SAA inference studies.

Thin, matplotlib-free drivers for the repeated-solve studies behind the
confidence-interval and limit-theorem algorithms in
:mod:`ensemblecontrol.inference`:

  * a sample-size *sweep* -- solve each nested prefix once, then form the plug-in,
    out-of-sample, and subsampling records (:func:`solve_saa_prefixes`,
    :func:`plugin_sweep`, :func:`plugin_oos_sweep`, :func:`subsampling_sweep`);
  * the CLT *replication study* -- ``R`` independent SAA solves per sample size
    (:func:`clt_replication_study`).

Everything is parameterized by a problem-agnostic solve callback

    solve(samples, w0=None, inner_serial=False) -> (saa, w_opt, f_opt)

and any :mod:`ensemblecontrol.sampling` sampler (needs ``.spawn`` and ``.sample``),
so a new problem reuses the whole pipeline by supplying only those two. Use
:func:`make_scipy_solve` for the common single-shooting / scipy-L-BFGS-B case (it
also serializes construction so it is safe inside the threaded studies).

The independent-solve loops share the outer/inner parallelism budget of
:func:`ensemblecontrol.inference._resolve_workers`: ``workers=1`` (default) is the
sequential loop with the inner per-sample CasADi map left threaded -- byte-for-byte
the prior behavior; ``workers="auto"`` adds outer threads *only* when a single
solve is light enough not to already saturate the cores (the two levels never
multiply), and ``inner_serial`` is set on the callback for those threaded solves.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .inference import (_BUILD_LOCK, _resolve_workers,
                        plugin_confidence_interval,
                        plugin_oos_confidence_interval,
                        subsampling_confidence_interval)

__all__ = [
    "make_scipy_solve", "make_ipopt_solve",
    "default_subsample_size", "default_num_subsamples",
    "solve_saa_prefixes", "plugin_sweep", "plugin_oos_sweep",
    "subsampling_sweep", "clt_replication_study", "coverage_study",
]


# -- reusable solve callback --------------------------------------------------

def make_scipy_solve(model, tol=1e-5, method="L-BFGS-B", verbose=False,
                     multiple_shooting=False, **saa_kwargs):
    """A ready-made ``solve(samples, w0=None, inner_serial=False)`` callback.

    Builds a single-shooting (by default) :class:`SAAProblem` and solves its
    control box with :class:`ScipyBoxSolver` (L-BFGS-B), returning
    ``(saa, w_opt, f_opt)`` -- the contract the studies here expect. Extra
    ``saa_kwargs`` (e.g. ``tol=`` for the SAA integrator, ``steps_per_interval=``)
    pass through to ``SAAProblem``.

    Construction happens under :data:`ensemblecontrol.inference._BUILD_LOCK` and
    the solve runs *outside* it, so the callback is safe to run concurrently
    inside :func:`clt_replication_study` (the heavy solves overlap; only the
    CasADi graph build is serialized). ``inner_serial=True`` builds the inner
    per-sample map serial (``parallelization="serial", n_threads=1``) -- the
    studies set it for solves that are themselves running under outer threads, so
    the two parallelism levels do not oversubscribe the cores.
    """
    from .saa_problem import SAAProblem
    from .scipy_box import ScipyBoxSolver

    def solve(samples, w0=None, inner_serial=False):
        kw = dict(saa_kwargs)
        if inner_serial:
            kw.update(parallelization="serial", n_threads=1)
        with _BUILD_LOCK:
            saa = SAAProblem(model, samples, MultipleShooting=multiple_shooting,
                             **kw)
            solver = ScipyBoxSolver(saa, method=method, tol=tol, verbose=verbose)
        w_opt, f_opt = solver.solve(w0=w0)
        return saa, w_opt, float(f_opt)

    return solve


def make_ipopt_solve(model, ipopt_options=None, tol=None, multiple_shooting=False,
                     warm_start_frac=0.01, **saa_kwargs):
    """A ready-made ``solve(samples, w0=None, inner_serial=False)`` callback using
    Ipopt.

    Builds a single-shooting (by default) :class:`SAAProblem` and solves it with
    ``SAAProblem.solve()`` (Ipopt via CasADi ``nlpsol``), returning
    ``(saa, w_opt, f_opt)`` -- the same contract as :func:`make_scipy_solve`, so
    the studies here can use either backend interchangeably. Pass ``ipopt_options``
    (e.g. tolerance + Hessian mode) and/or ``tol``; extra ``saa_kwargs`` pass
    through to ``SAAProblem``.

    When ``w0`` is given, the solve is warm-started from its controls, first
    clipped strictly inside the control box by ``warm_start_frac`` of each
    dimension's width. Ipopt is an interior-point method, so a bang-bang control
    sitting exactly on a bound is not a feasible interior start; the clip keeps the
    initial guess strictly interior (and works even when a bound is 0).

    Construction happens under :data:`_BUILD_LOCK` and the solve runs *outside* it,
    so the callback is safe to run concurrently inside :func:`clt_replication_study`;
    ``inner_serial=True`` pins the inner per-sample map serial so the outer and
    inner parallelism never oversubscribe the cores.
    """
    from .saa_problem import SAAProblem

    lb = np.asarray(model.control_bounds[0], dtype=float)
    ub = np.asarray(model.control_bounds[1], dtype=float)
    margin = warm_start_frac * (ub - lb)

    def solve(samples, w0=None, inner_serial=False):
        kw = dict(saa_kwargs)
        if inner_serial:
            kw.update(parallelization="serial", n_threads=1)
        with _BUILD_LOCK:
            saa = SAAProblem(model, samples, MultipleShooting=multiple_shooting,
                             ipopt_options=ipopt_options, tol=tol, **kw)
            if w0 is not None:
                controls = np.asarray(saa.control_matrix(w0), dtype=float)
                controls = np.clip(controls, lb + margin, ub - margin)
                saa.initial_decisions = saa.initial_from_controls(controls)
        w_opt, f_opt = saa.solve()
        # SAAProblem.solve returns f_opt as a length-1 array (Ipopt), not a scalar.
        return saa, w_opt, float(np.ravel(f_opt)[0])

    return solve


# -- default subsampling schedule (shared by the inference demos) -------------

def default_subsample_size(N):
    """Default subsampling block size ``b = floor(N^{6/7})`` (grows, ``b/N -> 0``).

    The ``+1e-9`` guards float rounding at exact powers (``128^{6/7} = 64``).
    """
    return int(np.floor(N ** (6.0 / 7.0) + 1e-9))


def default_num_subsamples(Nmax):
    """Default subsample count ``m = 5*Nmax``, held constant across the sweep."""
    return 5 * int(Nmax)


# -- sample-size sweep --------------------------------------------------------

def solve_saa_prefixes(solve, samples, sample_sizes):
    """Solve the SAA on each nested prefix ``samples[:N]`` once.

    ``solve(samples, w0=None) -> (saa, w_opt, f_opt)``. The nested prefixes reuse
    the same scenario draw, and each size-N solve is the anchor J_hat_N* / u_hat_N
    shared by every algorithm (so subsampling adds only its own re-solves, no
    extra size-N solve). Returns ``{N: (saa, w_opt, f_opt)}``.
    """
    return {N: solve(samples[:N]) for N in sample_sizes}


def plugin_sweep(solves, sample_sizes, levels=(0.90, 0.95, 0.99)):
    """Plug-in CI records (Algorithm 1) across the sample-size sweep.

    ``solves`` is the ``{N: (saa, w_opt, f_opt)}`` map from
    :func:`solve_saa_prefixes`. Returns the list of records to persist with
    :func:`ensemblecontrol.save_plugin_run`.
    """
    return [plugin_confidence_interval(solves[N][0], solves[N][1],
                                       f_opt=solves[N][2], levels=levels)
            for N in sample_sizes]


def plugin_oos_sweep(solves, sample_sizes, oos_problem,
                     levels=(0.90, 0.95, 0.99)):
    """Out-of-sample-s.d. plug-in CI records across the sweep.

    ``oos_problem(N) -> SAAProblem`` supplies the fresh, independent
    out-of-sample problem evaluated at each training size N -- the fixed-M and
    matched-M (M = N) variants are just different ``oos_problem`` callables.
    """
    return [plugin_oos_confidence_interval(solves[N][0], solves[N][1],
                                           oos_problem(N), f_opt=solves[N][2],
                                           levels=levels)
            for N in sample_sizes]


def subsampling_sweep(solves, sample_sizes, b_of=default_subsample_size,
                      m=None, seed=1, levels=(0.90, 0.95, 0.99), workers=1,
                      resolve_for=None, progress=False):
    """Subsampling CI records (Algorithm 2) across the sample-size sweep.

    ``b_of(N) -> int`` is the per-N block size (default ``floor(N^{6/7})``); ``m``
    defaults to ``default_num_subsamples(max(sample_sizes))``. Independent
    subsample-index streams are spawned per N from ``seed`` (numpy
    ``SeedSequence`` splitting), so the sweep is reproducible. ``workers`` is
    forwarded to each :func:`subsampling_confidence_interval` -- the parallelism
    is over the ``m`` re-solves within a single N, while the handful of sample
    sizes stay sequential. ``resolve_for(N, saa, w_opt) -> resolve`` optionally
    supplies a custom resolver per N (e.g. Ipopt); the default uses the built-in
    scipy L-BFGS-B resolver. Returns the list of records for
    :func:`ensemblecontrol.save_subsampling_run`.
    """
    sizes = list(sample_sizes)
    if m is None:
        m = default_num_subsamples(max(sizes))
    for N in sizes:                       # fail fast before any solve
        b = b_of(N)
        if not (0 < b < N):
            raise ValueError("subsample size b = {} must satisfy 0 < b < N = {}"
                             .format(b, N))
    streams = np.random.SeedSequence(seed).spawn(len(sizes))
    records = []
    for N, ss in zip(sizes, streams):
        saa, w_opt, f_opt = solves[N]
        resolve = None if resolve_for is None else resolve_for(N, saa, w_opt)
        records.append(subsampling_confidence_interval(
            saa, f_opt, b=b_of(N), m=m, rng=np.random.default_rng(ss),
            w_opt=w_opt, levels=levels, workers=workers, resolve=resolve,
            progress=progress))
    return records


# -- limit-theorem replication study ------------------------------------------

def clt_replication_study(sampler, solve, sample_sizes, R, n_ref,
                          warm_start=True, workers=1, progress=None):
    """Monte-Carlo replication study behind the SAA limit theorem.

    For each N in ``sample_sizes`` solve ``R`` independent SAA problems (fresh
    i.i.d. scenario draws) and record J_hat_N*; the statistic
    ``sqrt(N)(J_hat_N* - J_hat_ref*)`` is formed at plot time. J* is proxied by
    J_hat_ref*, the SAA value on an independent reference sample of size ``n_ref``.

    ``sampler`` is any i.i.d. ensemblecontrol sampler (the ROOT, before spawning):
    it is split into ``1 + len(sample_sizes)`` independent streams (reference plus
    one group per N), and each group spawns ``R`` replicate streams.
    ``solve(samples, w0=None, inner_serial=False) -> (saa, w_opt, f_opt)`` solves
    one SAA (see :func:`make_scipy_solve`); with ``warm_start`` every replicate
    starts from the reference control ``w_ref``.

    ``workers`` parallelizes the ``R`` replicate solves per N via the shared
    outer/inner budget (:func:`_resolve_workers` with ``inner_work=N``): ``1``
    (default) is the sequential, inner-threaded loop; ``"auto"`` adds outer
    threads only when a single size-N solve does not already saturate the cores,
    in which case the replicate solves are built ``inner_serial``. The reference
    solve always runs with the inner map threaded (one large solve).
    ``progress(N, done, total)`` is an optional per-N progress callback.

    Returns ``{"values_by_N": {N: ndarray[R]}, "f_ref": float, "w_ref": ndarray,
    "n_ref": int, "q": int}`` -- pass straight to
    :func:`ensemblecontrol.save_clt_run` (``q = len(w_ref)`` = control-mesh size).
    """
    sizes = list(sample_sizes)
    ref_sampler, *group_samplers = sampler.spawn(1 + len(sizes))

    _, w_ref, f_ref = solve(ref_sampler.sample(n_ref))   # inner map threaded
    w_ref = np.asarray(w_ref, dtype=float)
    f_ref = float(f_ref)
    w0 = w_ref if warm_start else None

    values_by_N = {}
    for N, gss in zip(sizes, group_samplers):
        rep_samplers = gss.spawn(R)
        # Pre-sample all R scenario arrays sequentially: rep_samplers are already
        # independent streams, so this is bit-identical to the sequential loop and
        # decouples the draw from the (possibly threaded) solve order.
        rep_samples = [rep_samplers[r].sample(N) for r in range(R)]
        n_workers = _resolve_workers(workers, R, inner_work=N)
        inner_serial = n_workers > 1
        report = ((lambda done: progress(N, done, R)) if callable(progress)
                  else (lambda done: None))

        def _value(samples):
            _, _, f = solve(samples, w0=w0, inner_serial=inner_serial)
            return float(f)

        vals = np.empty(R)
        if n_workers == 1:
            for r, samples in enumerate(rep_samples):
                vals[r] = _value(samples)
                report(r + 1)
        else:
            # ThreadPoolExecutor.map preserves input order -> vals[r] stays aligned
            # to rep_samples (identical to the sequential result).
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                for r, f in enumerate(ex.map(_value, rep_samples)):
                    vals[r] = f
                    report(r + 1)
        values_by_N[N] = vals

    return {"values_by_N": values_by_N, "f_ref": f_ref, "w_ref": w_ref,
            "n_ref": int(n_ref), "q": int(w_ref.size)}


# -- coverage test ------------------------------------------------------------

def coverage_study(sampler, solve, sample_sizes, R, n_ref,
                   levels=(0.90, 0.95, 0.99), ci_of=None, warm_start=True,
                   workers=1, progress=None):
    """Monte-Carlo coverage test for the SAA confidence intervals.

    For each N in ``sample_sizes`` run ``R`` independent replications: draw a
    fresh i.i.d. sample of size N, solve the SAA, build a confidence interval, and
    record whether it covers the reference value J_hat_ref* -- the population
    optimal J*, proxied by the SAA value on one independent reference sample of
    size ``n_ref`` (the same proxy across all N, since J* does not depend on the
    training size).  The per-level coverage indicators are the raw Bernoulli data
    behind :func:`ensemblecontrol.probability_lower_bound`; aggregate them with
    :func:`ensemblecontrol.coverage_from_indicators` /
    :func:`ensemblecontrol.coverage_latex_table`.

    ``sampler`` is the ROOT i.i.d. sampler, split into ``1 + len(sample_sizes)``
    independent streams (reference plus one group per N, each spawning ``R``
    replicate streams).  ``solve(samples, w0=None, inner_serial=False) -> (saa,
    w_opt, f_opt)`` solves one SAA (see :func:`make_scipy_solve` /
    :func:`make_ipopt_solve`); with ``warm_start`` every replicate starts from the
    reference control ``w_ref``.  ``ci_of(saa, w_opt, f_opt) -> ci`` builds the
    interval; the default is the plug-in CI (:func:`plugin_confidence_interval`,
    one extra rollout -- no re-solve -- so the cost is exactly R+1 solves per N),
    guarded by ``_BUILD_LOCK`` because building it constructs CasADi graphs.  A
    custom ``ci_of`` that builds CasADi objects must likewise guard construction
    with ``_BUILD_LOCK`` (importable from :mod:`ensemblecontrol.inference`) and stay
    internally serial when ``workers > 1``; a subsampling ``ci_of`` would cost
    ``R*m`` solves per N.

    ``workers`` parallelizes the ``R`` replicate solves per N via the shared
    outer/inner budget (:func:`_resolve_workers` with ``inner_work=N``), exactly as
    in :func:`clt_replication_study`: ``1`` (default) is the sequential,
    inner-threaded loop; ``"auto"`` adds outer threads only when a single size-N
    solve does not already saturate the cores, in which case the replicate solves
    are built ``inner_serial``.  The reference solve always runs inner-threaded.
    ``progress(N, done, R)`` is an optional per-N progress callback.

    Returns ``{"sample_sizes", "n_ref", "f_ref", "w_ref", "q", "R", "levels",
    "indicators_by_N": {N: {level: ndarray[bool]}}, "bounds_by_N": {N: {level:
    {"lo": ndarray, "hi": ndarray}}}}`` -- pass straight to
    :func:`ensemblecontrol.save_coverage_run`.
    """
    sizes = list(sample_sizes)
    levels = tuple(levels)
    ref_sampler, *group_samplers = sampler.spawn(1 + len(sizes))

    _, w_ref, f_ref = solve(ref_sampler.sample(n_ref))   # inner map threaded
    w_ref = np.asarray(w_ref, dtype=float)
    f_ref = float(f_ref)
    w0 = w_ref if warm_start else None

    if ci_of is None:
        def ci_of(saa, w_opt, f_opt):
            # terminal_losses rebuilds CasADi graphs on each call -> serialize.
            with _BUILD_LOCK:
                rec = plugin_confidence_interval(saa, w_opt, f_opt=f_opt,
                                                 levels=levels)
            return rec["ci"]

    indicators_by_N = {}
    bounds_by_N = {}
    for N, gss in zip(sizes, group_samplers):
        rep_samplers = gss.spawn(R)
        # Pre-sample sequentially (independent streams) so the indicators are
        # bit-identical regardless of the (possibly threaded) solve order.
        rep_samples = [rep_samplers[r].sample(N) for r in range(R)]
        n_workers = _resolve_workers(workers, R, inner_work=N)
        inner_serial = n_workers > 1
        report = ((lambda done: progress(N, done, R)) if callable(progress)
                  else (lambda done: None))

        def _row(samples):
            saa, w_opt, f_opt = solve(samples, w0=w0, inner_serial=inner_serial)
            ci = ci_of(saa, w_opt, f_opt)
            lo = np.array([ci["levels"][lvl]["lo"] for lvl in levels])
            hi = np.array([ci["levels"][lvl]["hi"] for lvl in levels])
            return lo, hi

        lo_mat = np.empty((R, len(levels)))
        hi_mat = np.empty((R, len(levels)))
        if n_workers == 1:
            for r, samples in enumerate(rep_samples):
                lo_mat[r], hi_mat[r] = _row(samples)
                report(r + 1)
        else:
            # ThreadPoolExecutor.map preserves input order -> rows stay aligned.
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                for r, (lo, hi) in enumerate(ex.map(_row, rep_samples)):
                    lo_mat[r], hi_mat[r] = lo, hi
                    report(r + 1)

        indicators_by_N[N] = {}
        bounds_by_N[N] = {}
        for j, lvl in enumerate(levels):
            lo = lo_mat[:, j]
            hi = hi_mat[:, j]
            indicators_by_N[N][lvl] = (lo <= f_ref) & (f_ref <= hi)
            bounds_by_N[N][lvl] = {"lo": lo.copy(), "hi": hi.copy()}

    return {"sample_sizes": sizes, "n_ref": int(n_ref), "f_ref": f_ref,
            "w_ref": w_ref, "q": int(w_ref.size), "R": int(R),
            "levels": list(levels), "indicators_by_N": indicators_by_N,
            "bounds_by_N": bounds_by_N}
