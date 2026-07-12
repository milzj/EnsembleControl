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
    "monotonicity_check", "format_monotonicity_table",
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


def default_num_subsamples(N):
    """Default per-N subsample count ``m_N = 5N`` (grows with the sample size)."""
    return 5 * int(N)


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

    ``b_of(N) -> int`` is the per-N block size (default ``floor(N^{6/7})``). ``m`` is
    the subsample count -- a per-N callable ``m_of(N) -> int`` or a scalar (constant
    across the sweep); it defaults to the per-N rule ``m_N = 5N``
    (:func:`default_num_subsamples`). Independent
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
    # m: a per-N callable m_of(N) -> int, or a scalar (constant across the sweep).
    # Default (m is None) is the per-N rule m_N = 5N (default_num_subsamples).
    if m is None:
        m_of = default_num_subsamples
    elif callable(m):
        m_of = m
    else:
        m_of = lambda N: m
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
            saa, f_opt, b=b_of(N), m=m_of(N), rng=np.random.default_rng(ss),
            w_opt=w_opt, levels=levels, workers=workers, resolve=resolve,
            progress=progress))
    return records


# -- limit-theorem replication study ------------------------------------------

def clt_replication_study(sampler, solve, sample_sizes, R, n_ref,
                          warm_start=True, workers=1, progress=None):
    """Monte-Carlo replication study behind the SAA limit theorem.

    For each N in ``sample_sizes`` record ``R`` replicate SAA optimal values J_hat_N*;
    the statistic ``sqrt(N)(J_hat_N* - J_hat_ref*)`` is formed at plot time. J* is
    proxied by J_hat_ref*, the SAA value on an independent reference sample of size
    ``n_ref``.

    The R replicates use COMMON RANDOM NUMBERS across N (the canonical SAA
    construction): replicate ``r`` draws ``max(sample_sizes)`` scenarios once and its
    size-N value uses the nested prefix ``samples[:N]`` -- so within a replicate the
    size-32 problem is literally the first 32 scenarios of its size-64 problem.
    Independent replicates draw independent sequences.

    ``sampler`` is any i.i.d. ensemblecontrol sampler (the ROOT, before spawning): it
    is split into ``1 + R`` independent streams -- one reference plus one per replicate.
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
    sizes = sorted({int(n) for n in sample_sizes})
    n_max = sizes[-1]
    ref_sampler, *rep_samplers = sampler.spawn(1 + R)

    _, w_ref, f_ref = solve(ref_sampler.sample(n_ref))   # inner map threaded
    w_ref = np.asarray(w_ref, dtype=float)
    f_ref = float(f_ref)
    w0 = w_ref if warm_start else None

    # Common random numbers across N: replicate r draws n_max scenarios ONCE (upfront,
    # decoupled from the possibly-threaded solve order), and its size-N value uses the
    # nested prefix full_samples[r][:N] -- the canonical SAA construction, so within a
    # replicate the size-32 problem is the first 32 scenarios of its size-64 problem.
    full_samples = [rep_samplers[r].sample(n_max) for r in range(R)]

    values_by_N = {}
    for N in sizes:
        rep_samples = [full[:N] for full in full_samples]
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
                   workers=1, progress=None, ref_solve=None):
    """Monte-Carlo coverage test for the SAA confidence intervals.

    For each N in ``sample_sizes`` run ``R`` replications: form a size-N sample,
    solve the SAA, build a confidence interval, and record whether it covers the
    reference value J_hat_ref* -- the population optimal J*, proxied by the SAA
    value on one independent reference sample of size ``n_ref`` (the same proxy
    across all N, since J* does not depend on the training size).  The per-level
    coverage indicators are the raw Bernoulli data behind
    :func:`ensemblecontrol.probability_lower_bound`; aggregate them with
    :func:`ensemblecontrol.coverage_from_indicators` /
    :func:`ensemblecontrol.coverage_latex_table`.

    The R replications use COMMON RANDOM NUMBERS across N (the canonical SAA
    construction, as in :func:`clt_replication_study`): replicate ``r`` draws
    ``max(sample_sizes)`` scenarios once and its size-N interval is built on the
    nested prefix ``samples[:N]`` -- so within a replicate the size-32 problem is
    literally the first 32 scenarios of its size-64 problem.  The reference stream
    stays independent of every training sample: the CI must be tested against a J*
    proxy independent of the data it is built from, or the coverage estimate is
    biased -- only the training samples across N are nested, never the reference.

    ``sampler`` is the ROOT i.i.d. sampler, split into ``1 + R`` independent
    streams -- one reference plus one per replicate.
    ``solve(samples, w0=None, inner_serial=False) -> (saa,
    w_opt, f_opt)`` solves one SAA (see :func:`make_scipy_solve` /
    :func:`make_ipopt_solve`); with ``warm_start`` every replicate starts from the
    reference control ``w_ref``.  ``ci_of(saa, w_opt, f_opt) -> ci`` builds the
    interval; the default is the plug-in CI (:func:`plugin_confidence_interval`,
    one extra rollout -- no re-solve -- so the cost is exactly R+1 solves per N),
    guarded by ``_BUILD_LOCK`` because building it constructs CasADi graphs.
    ``ref_solve`` (default ``solve``) solves the single reference sample; pass a
    tighter-tolerance callback than the replicate ``solve`` so J_hat_ref* matches
    another study's reference exactly (e.g. the CLT study's J_hat_ref*).  A
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
    sizes = sorted({int(n) for n in sample_sizes})
    n_max = sizes[-1]
    levels = tuple(levels)
    ref_sampler, *rep_samplers = sampler.spawn(1 + R)

    # Reference J*_{n_ref} proxy: solved with ref_solve (default: solve) so a study
    # can compute it at a tighter tolerance than its many replicate solves, keeping
    # J_hat_ref* identical to another study's reference (e.g. the CLT study's).
    ref_solve = ref_solve or solve
    _, w_ref, f_ref = ref_solve(ref_sampler.sample(n_ref))   # inner map threaded
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

    # Common random numbers across N: replicate r draws n_max scenarios ONCE (upfront,
    # decoupled from the possibly-threaded solve order), and its size-N interval is
    # built on the nested prefix full_samples[r][:N] -- so within a replicate the
    # size-32 problem is the first 32 scenarios of its size-64 problem. The reference
    # sample above is drawn from an independent stream (unbiased coverage target).
    full_samples = [rep_samplers[r].sample(n_max) for r in range(R)]

    indicators_by_N = {}
    bounds_by_N = {}
    for N in sizes:
        rep_samples = [full[:N] for full in full_samples]
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


# -- monotonicity diagnostic for the SAA optimal-value means ------------------

def _values_by_N(run_or_values):
    """Coerce a study result / loaded run / plain mapping to ``{int N: ndarray[R]}``."""
    if "values_by_N" in run_or_values:
        src = run_or_values["values_by_N"]
    elif "results" in run_or_values:
        src = {rec["N"]: rec["values"] for rec in run_or_values["results"]}
    else:
        src = run_or_values
    return {int(N): np.asarray(v, dtype=float) for N, v in src.items()}


def monotonicity_check(run_or_values, nested=True):
    """Adjacent-difference monotonicity diagnostic for the SAA optimal-value means.

    For a minimization the theoretical means ``m_N = E[Jhat_N*]`` are nondecreasing in
    N (optimistic bias, Prop. 5.6), but the Monte-Carlo estimates
    ``mhat_N = mean_r Jhat_N*^(r)`` need not be, because of finite-R noise.  For each
    adjacent pair N1 < N2 this returns the estimated gap ``delta = mhat_N2 - mhat_N1``,
    its standard error, the z-score ``delta / se``, and a status label separating
    harmless MC noise from a genuine violation.  The aim is diagnosis -- NOT forcing the
    estimated curve to be monotone.

    ``nested=True`` is the design used by :func:`clt_replication_study` and
    :func:`optimal_value_study` (replicate r draws max(N) scenarios once and size-N uses
    the nested prefix), so the two sizes are paired replicate-by-replicate and
    ``se = std(Jhat_N2^(r) - Jhat_N1^(r), ddof=1)/sqrt(R)`` -- the correct, much tighter
    standard error for the difference of two positively-correlated means.  It requires
    equal R across sizes (raises ``ValueError`` otherwise).  ``nested=False`` treats the
    sizes as independent and uses ``se = sqrt(se(mhat_N2)^2 + se(mhat_N1)^2)``.

    ``run_or_values`` may be a study result (has ``values_by_N``), a loaded run (has
    ``results``), or a plain ``{N: values}`` mapping.  Returns one row dict per adjacent
    pair with keys ``N1, N2, m1, m2, delta, se, z, status``; ``status`` is ``"OK"``
    (delta >= 0), ``"compatible with MC noise"`` (delta < 0 but z >= -2), or
    ``"potential issue"`` (z < -2).
    """
    vals = _values_by_N(run_or_values)
    sizes = sorted(vals)
    rows = []
    for N1, N2 in zip(sizes[:-1], sizes[1:]):
        v1, v2 = vals[N1], vals[N2]
        m1, m2 = float(v1.mean()), float(v2.mean())
        delta = m2 - m1
        if nested:
            if v1.size != v2.size:
                raise ValueError(
                    "nested=True needs equal R across sizes (N=%d has R=%d, N=%d has "
                    "R=%d); pass nested=False for independent samples"
                    % (N1, v1.size, N2, v2.size))
            D = v2 - v1
            se = float(D.std(ddof=1) / np.sqrt(D.size)) if D.size > 1 else 0.0
        else:
            se1 = v1.std(ddof=1) / np.sqrt(v1.size) if v1.size > 1 else 0.0
            se2 = v2.std(ddof=1) / np.sqrt(v2.size) if v2.size > 1 else 0.0
            se = float(np.hypot(se1, se2))
        if se > 0:
            z = delta / se
        else:
            z = 0.0 if delta == 0 else float(np.copysign(np.inf, delta))
        if delta >= 0:
            status = "OK"
        elif z >= -2:
            status = "compatible with MC noise"
        else:
            status = "potential issue"
        rows.append({"N1": N1, "N2": N2, "m1": m1, "m2": m2,
                     "delta": delta, "se": se, "z": z, "status": status})
    return rows


def format_monotonicity_table(rows):
    """Plain-text table for :func:`monotonicity_check` rows.

    Columns: ``N1, N2, m_N1, m_N2, Delta, se(Delta), z, status``.  Returns a string.
    """
    header = ("%4s %4s %13s %13s %12s %11s %8s  %s"
              % ("N1", "N2", "m_N1", "m_N2", "Delta", "se(Delta)", "z", "status"))
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append("%4d %4d %13.6f %13.6f %12.6f %11.6f %8.2f  %s"
                     % (r["N1"], r["N2"], r["m1"], r["m2"], r["delta"],
                        r["se"], r["z"], r["status"]))
    return "\n".join(lines)
