"""Monte-Carlo estimate of the expected SAA optimal value E[Jhat_N*] vs sample size N.

For each N in ``sample_sizes`` we form ``R`` SAA optimal values Jhat_N* and average them
to estimate E[Jhat_N*].  The R replicates use **common random numbers across N**:
replicate ``r`` draws ``max(N)`` scenarios once and every size-N solve reuses the nested
prefix ``samples[:N]``, so the only thing that changes with N is how many of the *same*
draws enter the sample average.  This is the natural SAA design -- it makes the E[Jhat_N*]
estimates positively correlated across N, so the monotone increase (Prop. 5.6:
E[Jhat_N*] <= E[Jhat_{N+1}*] <= J*) shows through with far less Monte-Carlo noise than
independent draws would give.

Unlike the CLT study (:func:`ensemblecontrol.clt_replication_study`) this needs **no
reference solve**: a plot of the mean optimal value +/- SE against N is a convergence /
sanity diagnostic on its own -- there is no J* proxy to estimate or subtract.

Everything is parameterized by the same problem-agnostic solve callback the inference
studies use::

    solve(samples, w0=None, inner_serial=False) -> (saa, w_opt, f_opt)

and any :mod:`ensemblecontrol.sampling` sampler (needs ``.spawn`` and ``.sample``), so a
new problem reuses the pipeline by supplying only those two.  Results persist to JSON via
:func:`save_optimal_value_run` and re-plot from it with :func:`plot_optimal_value`, so the
figure can be redrawn without re-running the R*len(N) solves.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .inference import _resolve_workers, _created, _write_json

__all__ = [
    "optimal_value_study",
    "mean_value_series",
    "save_optimal_value_run",
    "load_optimal_value_run",
    "plot_optimal_value",
]


def _solve_group(solve, rep_samples, N, R, workers, progress):
    """Solve the R replicates for one sample size N; return (values[R], q).

    Shares the outer/inner parallelism budget of :func:`_resolve_workers` exactly as
    :func:`ensemblecontrol.clt_replication_study` does: ``workers=1`` is the sequential
    inner-threaded loop, ``"auto"`` adds outer threads only when a single size-N solve does
    not already saturate the cores (then the inner map is pinned serial).
    """
    n_workers = _resolve_workers(workers, R, inner_work=N)
    inner_serial = n_workers > 1
    report = ((lambda done: progress(N, done, R)) if callable(progress)
              else (lambda done: None))
    q_box = {}

    def _value(samples):
        _, w_opt, f = solve(samples, w0=None, inner_serial=inner_serial)
        q_box.setdefault("q", int(np.asarray(w_opt).size))   # mesh size (same for all)
        return float(f)

    vals = np.empty(R)
    if n_workers == 1:
        for r, samples in enumerate(rep_samples):
            vals[r] = _value(samples)
            report(r + 1)
    else:
        # ThreadPoolExecutor.map preserves input order -> vals[r] stays aligned to
        # rep_samples (byte-identical to the sequential result).
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for r, f in enumerate(ex.map(_value, rep_samples)):
                vals[r] = f
                report(r + 1)
    return vals, q_box.get("q")


def optimal_value_study(sampler, solve, sample_sizes, R, workers=1, progress=None):
    """Estimate E[Jhat_N*] by ``R`` replicate SAA solves per N, with common random
    numbers across N (no reference solve).

    ``sampler`` is a ROOT ensemblecontrol sampler; it is split into ``R`` independent
    replicate streams.  Replicate ``r`` draws ``max(sample_sizes)`` scenarios **once**, and
    its size-N optimal value uses the nested prefix ``samples[:N]`` -- so within a replicate
    the size-32 problem is literally the first 32 scenarios of its size-64 problem (common
    random numbers).  ``solve(samples, w0=None, inner_serial=False) -> (saa, w_opt, f_opt)``
    solves one SAA (its own start; ``w0`` is left ``None`` here).  ``workers`` shares the
    outer/inner budget of :func:`ensemblecontrol.inference._resolve_workers`.
    ``progress(N, done, total)`` is an optional per-N callback.

    Returns ``{"values_by_N": {N: ndarray[R]}, "q": int}`` (``q`` = control-mesh size, read
    off a solved decision vector).
    """
    sizes = sorted({int(n) for n in sample_sizes})
    n_max = sizes[-1]
    # R independent replicate streams; each draws n_max scenarios once. The size-N solve
    # for replicate r reuses the nested prefix full_samples[r][:N] -- common random numbers
    # across N, the crux of SAA (only the count entering the sample average changes with N).
    rep_samplers = sampler.spawn(R)
    full_samples = [rep_samplers[r].sample(n_max) for r in range(R)]

    values_by_N = {}
    q = None
    for N in sizes:
        prefixes = [full[:N] for full in full_samples]
        vals, q_i = _solve_group(solve, prefixes, N, R, workers, progress)
        values_by_N[N] = vals
        if q is None:
            q = q_i

    return {"values_by_N": values_by_N, "q": q}


def mean_value_series(run):
    """Sorted ``(N, mean, se)`` arrays from a study result or a loaded run.

    Accepts either the dict from :func:`optimal_value_study` (has ``values_by_N``) or one
    from :func:`load_optimal_value_run` (has ``results``).  ``se = std(ddof=1)/sqrt(R)``
    treats each E[Jhat_N*] as an ordinary sample mean over the R replicates.
    """
    if "results" in run:
        rows = sorted(run["results"], key=lambda rec: rec["N"])
        Ns = [rec["N"] for rec in rows]
        vals = [np.asarray(rec["values"], dtype=float) for rec in rows]
    else:
        Ns = sorted(run["values_by_N"])
        vals = [np.asarray(run["values_by_N"][n], dtype=float) for n in Ns]
    N = np.asarray(Ns, dtype=float)
    mean = np.array([v.mean() for v in vals])
    se = np.array([v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0
                   for v in vals])
    return N, mean, se


def save_optimal_value_run(run, path, meta=None, r=None):
    """Persist an optimal-value study to ``path`` as JSON (mirrors
    :func:`ensemblecontrol.save_clt_run`, minus the reference).  ``q`` is the control-mesh
    size and ``r`` the scenario radius (both listed in the plot legend).  Returns ``path``.
    """
    values_by_N = (run["values_by_N"] if "values_by_N" in run
                   else {rec["N"]: rec["values"] for rec in run["results"]})
    data = {
        "algorithm": "optimal_value",
        "q": run.get("q"),
        "r": r,
        "results": [{"N": int(N),
                     "values": np.asarray(values_by_N[N], dtype=float).tolist()}
                    for N in sorted(values_by_N)],
        "meta": _created(meta),
    }
    _write_json(path, data)
    return path


def load_optimal_value_run(path):
    """Load a run saved by :func:`save_optimal_value_run` (values as ndarrays)."""
    with open(path) as fh:
        data = json.load(fh)
    for rec in data["results"]:
        rec["values"] = np.asarray(rec["values"], dtype=float)
    return data


def _qr_note(q, r):
    """One-line ``(q = .., r = ..)`` legend annotation from whichever is provided."""
    parts = []
    if q is not None:
        parts.append("q = {}".format(q))
    if r is not None:
        parts.append("r = {}".format(r))
    return (r"$(%s)$" % ", ".join(parts)) if parts else None


def plot_optimal_value(run_or_path, outdir=None, stamp=None, formats=("png",)):
    """Plot E[Jhat_N*] +/- SE vs sample size N (single panel; no reference line).

    ``run_or_path`` is a study/loaded run dict or a path to a saved JSON.  Writes
    ``mean_saa_optimal_value[_<stamp>].{formats}`` to ``outdir`` (defaults to the JSON's
    directory when a path is given).  Returns the list of written paths.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from .inference_plotting import configure_style

    if isinstance(run_or_path, str):
        run = load_optimal_value_run(run_or_path)
        outdir = outdir or os.path.dirname(run_or_path)
    else:
        run = run_or_path
    if outdir is None:
        raise ValueError("outdir is required when run_or_path is not a path")

    N, mean, se = mean_value_series(run)

    configure_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.errorbar(N, mean, yerr=se, fmt="o-", color="C0", capsize=3, zorder=3,
                label=r"$\mathbb{E}[\widehat J_N^*] \pm$ standard error")
    ax.set_xscale("log", base=2)
    ax.set_xticks(N)
    ax.set_xticklabels([("%d" % n) for n in N])
    ax.minorticks_off()
    ax.set_xlabel(r"sample size $N$")
    ax.set_ylabel(r"mean SAA optimal value $\mathbb{E}[\widehat J_N^*]$")

    handles, labels = ax.get_legend_handles_labels()
    note = _qr_note(run.get("q"), run.get("r"))
    if note is not None:
        handles.append(mpatches.Patch(color="none"))
        labels.append(note)
    ax.legend(handles, labels, loc="best")
    fig.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    stem = ("mean_saa_optimal_value" if not stamp
            else "mean_saa_optimal_value_%s" % stamp)
    base = os.path.join(outdir, stem)
    written = []
    for ext in formats:
        p = "%s.%s" % (base, ext)
        fig.savefig(p)
        written.append(p)
    plt.close(fig)
    return written
