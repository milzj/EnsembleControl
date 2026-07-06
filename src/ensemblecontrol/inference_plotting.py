"""Figures for the SAA inference algorithms, rendered from saved raw data.

Every ``plot_*`` function accepts either an in-memory run/record dict or a path
to a JSON file written by ``inference.save_*_run``, and recomputes the
confidence intervals from the raw data (per-scenario losses F_i / subsampling
statistics Delta_r).  No solving happens here, so figures can be restyled,
re-labelled, or re-banded without re-running the optimization -- e.g. run the
compute step once, then re-plot repeatedly with different legends or CI bands.

``plot_plugin`` reproduces the plug-in figures (per-N and combined F_i
histograms with the J_hat_N* line and shaded CI band; per-level J_hat_N*-vs-N
error bars; per-level half-width log-log 1/sqrt(N) scaling).  ``plot_subsampling``
draws the Delta_r histogram with the empirical CI quantiles marked.
"""

import os
import shutil
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy.stats import norm

from .inference import (plugin_ci_from_losses, plugin_oos_ci_from_losses,
                        subsampling_ci_from_deltas,
                        load_plugin_run, load_subsampling_run,
                        clt_statistic, load_clt_run)

__all__ = ["plot_plugin", "plot_subsampling", "plot_clt", "configure_style",
           "value_ylim_across"]

# Confidence levels drawn widest-first so the narrower (higher-opacity) bands sit
# on top; the increasing alpha makes the nesting 99 > 95 > 90 read directly.
_LEVEL_ALPHA = {0.90: 0.45, 0.95: 0.28, 0.99: 0.15}
_BAND_COLOR = "C1"

_DEFAULT_LOSS_LABEL = r"$F(x_{\widehat{u}_N}(t_f,\xi_i))$"
_DEFAULT_VALUE_LABEL = r"optimal value $\widehat J_N^*$"
_DEFAULT_DELTA_LABEL = r"$\sqrt{b}\,(J^*_{I_r} - \widehat J_N^*)$"

_STYLE_CONFIGURED = False


def configure_style():
    """Publication figure style, applied once per process.

    Uses LaTeX (serif Computer Modern) when a ``latex`` binary is on PATH,
    otherwise matplotlib's default mathtext so plots still render without TeX.
    """
    global _STYLE_CONFIGURED
    if _STYLE_CONFIGURED:
        return
    _STYLE_CONFIGURED = True
    style = {
        "lines.linewidth": 2,
        "font.size": 12.5,
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    }
    if shutil.which("latex"):
        style.update({
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsfonts}",
            "font.family": "serif",
            "font.serif": "Computer Modern Roman",
            "font.monospace": "Computer Modern Typewriter",
        })
    plt.rcParams.update(style)


def _pct(level):
    # "%" starts a comment in LaTeX, so escape it only when usetex is active.
    esc = r"\%" if plt.rcParams.get("text.usetex", False) else "%"
    return "{:.0f}{}".format(100 * level, esc)


def _stamp(stamp):
    # dash-separated, filename-safe: 2026-07-05T19-51-18
    return stamp if stamp is not None else datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def value_ylim_across(cis, margin=0.05):
    """Common ``(lo, hi)`` optimal-value y-range covering the intervals (and each
    J_hat_N*) of all confidence-interval dicts ``cis``, padded by fractional
    ``margin``.

    Pass the plug-in and subsampling CI dicts (record ``["ci"]`` values) so both
    families of ``_ci{level}`` interval plots can share a y-axis for comparison::

        ylim = value_ylim_across([r["ci"] for r in plugin_records]
                                 + [r["ci"] for r in sub_records])
        plot_plugin(..., value_ylim=ylim); plot_subsampling(..., value_ylim=ylim)
    """
    vals = []
    for ci in cis:
        vals.append(ci["Jhat"])
        for bnd in ci["levels"].values():
            vals.append(bnd["lo"])
            vals.append(bnd["hi"])
    lo, hi = min(vals), max(vals)
    pad = margin * (hi - lo) if hi > lo else (abs(hi) or 1.0)
    return (lo - pad, hi + pad)


def _labels(labels):
    labels = dict(labels) if labels else {}
    return (labels.get("loss_label", _DEFAULT_LOSS_LABEL),
            labels.get("value_label", _DEFAULT_VALUE_LABEL),
            labels.get("delta_label", _DEFAULT_DELTA_LABEL))


# -- run resolution ----------------------------------------------------------

def _as_plugin_run(run_or_path):
    if isinstance(run_or_path, str):
        return load_plugin_run(run_or_path)
    # accept an in-memory run dict, or a single/list of compute records
    if "results" in run_or_path:
        return run_or_path
    records = run_or_path if isinstance(run_or_path, (list, tuple)) else [run_or_path]

    def _res(r):
        d = {"N": r["N"], "f_opt": r["f_opt"], "F": np.asarray(r["F"], float)}
        if "M" in r:
            d["M"] = r["M"]
        return d
    return {"algorithm": "plugin",
            "variance": records[0].get("variance", "in-sample"),
            "levels": list(records[0]["levels"]),
            "q": records[0].get("q"),
            "results": [_res(r) for r in records]}


def _as_subsampling_run(run_or_path):
    if isinstance(run_or_path, str):
        return load_subsampling_run(run_or_path)
    run = run_or_path
    if "results" in run:
        return run
    # accept an in-memory single record or a list of records (one per N)
    records = run if isinstance(run, (list, tuple)) else [run]
    return {"algorithm": "subsampling",
            "levels": list(records[0]["levels"]),
            "q": records[0].get("q"),
            "results": [{"N": r["N"], "f_opt": r["f_opt"], "b": r["b"],
                         "m": r["m"], "deltas": np.asarray(r["deltas"], float)}
                        for r in records]}


# -- plug-in drawing helpers (ported from the fed-batch demo) ----------------

def _draw_ci_hist(ax, N, F, ci, band_levels, loss_label):
    ax.hist(F, bins="auto", color="0.8", edgecolor="0.4")
    for level in sorted(band_levels, reverse=True):   # widest first if several
        bnd = ci["levels"][level]
        ax.axvspan(bnd["lo"], bnd["hi"], color=_BAND_COLOR,
                   alpha=_LEVEL_ALPHA.get(level, 0.3))
        ax.axvline(bnd["lo"], color=_BAND_COLOR, lw=1.0, ls="--")
        ax.axvline(bnd["hi"], color=_BAND_COLOR, lw=1.0, ls="--")
    ax.axvline(ci["Jhat"], color="k", lw=1.5)
    ax.set_xlabel(loss_label)
    ax.set_title(r"$N = {}$".format(N))


def _q_handle(q):
    # Invisible legend entry annotating the control-mesh size q (= number of
    # control intervals = len(control)), matching the demo's CI figures.
    return mpatches.Patch(color="none", label=r"$(q = {})$".format(q))


def _append_q(handles, q):
    return handles if q is None else list(handles) + [_q_handle(q)]


def _hist_legend_handles(band_levels, loss_label):
    handles = [mpatches.Patch(facecolor="0.8", edgecolor="0.4",
                              label=r"$F_i$ histogram"),
               mlines.Line2D([], [], color="k", lw=1.5,
                             label=r"$\widehat J_N^*$")]
    for level in sorted(band_levels):
        handles.append(mpatches.Patch(
            color=_BAND_COLOR, alpha=_LEVEL_ALPHA.get(level, 0.3),
            label="{} CI".format(_pct(level))))
    return handles


def _draw_ci_errorbar(ax, cis, level, value_label):
    Ns = [ci["N"] for ci in cis]
    Jhat = [ci["Jhat"] for ci in cis]
    hw = [ci["levels"][level]["halfwidth"] for ci in cis]
    ax.errorbar(Ns, Jhat, yerr=hw, fmt="o-", capsize=5, color="C0",
                ecolor="C1", elinewidth=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(value_label)
    ax.set_title("{} confidence interval".format(_pct(level)))
    ax.grid(True, which="both", alpha=0.3)


def _draw_hw_loglog(ax, cis, level):
    Ns = np.array([ci["N"] for ci in cis], dtype=float)

    def _hw(ci):
        # plug-in stores an explicit symmetric half-width; the subsampling
        # interval is asymmetric, so use half the total width.
        bnd = ci["levels"][level]
        return bnd["halfwidth"] if "halfwidth" in bnd else 0.5 * (bnd["hi"] - bnd["lo"])
    hw = np.array([_hw(ci) for ci in cis], dtype=float)

    ax.plot(Ns, hw, "o-", color="C1", label="CI half-width")
    intercept = np.mean(np.log(hw) + 0.5 * np.log(Ns))
    ref = np.exp(intercept - 0.5 * np.log(Ns))
    ax.plot(Ns, ref, "k--", label=r"$\propto N^{-1/2}$")

    slope = float(np.polyfit(np.log(Ns), np.log(hw), 1)[0])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(int(n)) for n in Ns])
    ax.set_xlabel(r"$N$")
    ax.set_ylabel("CI half-width")
    ax.set_title(r"half-width scaling (fit {:.2f} vs. ideal $-0.5$)".format(slope))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()


# -- public: plug-in figures -------------------------------------------------

def plot_plugin(run_or_path, outdir=None, prefix="plugin", stamp=None,
                levels=None, hist_bands=(0.95,), q=None, labels=None,
                value_ylim=None):
    """Render the plug-in figures from a run (dict) or a saved JSON ``path``.

    Recomputes every confidence interval from the raw losses, so ``levels``,
    ``hist_bands`` and ``labels`` are pure display choices needing no recompute.
    ``q`` (the control-mesh size, = number of control intervals) is annotated on
    every figure's legend; it defaults to the value stored in the run.
    ``value_ylim=(lo, hi)`` fixes the optimal-value (y) axis of the per-level
    ``_ci{level}`` interval plots -- pass the same value to ``plot_subsampling``
    so the plug-in and subsampling intervals share a y-axis for comparison.
    When ``outdir`` is given, saves PNGs named ``{prefix}_{stamp}_...`` and
    returns the list of paths; otherwise returns the list of ``(fig, ax)``.
    """
    configure_style()
    run = _as_plugin_run(run_or_path)
    results = run["results"]
    levels = tuple(levels) if levels is not None else tuple(run["levels"])
    hist_bands = tuple(hist_bands)
    q = q if q is not None else run.get("q")
    loss_label, value_label, _ = _labels(labels)

    if run.get("variance") == "out-of-sample":
        cis = [plugin_oos_ci_from_losses(r["F"], r["f_opt"], r["N"], levels)
               for r in results]
    else:
        cis = [plugin_ci_from_losses(r["F"], r["f_opt"], levels) for r in results]

    # N_out: out-of-sample size M (only present for the out-of-sample variant);
    # listed in every legend alongside q. A fixed M prints the number; M matched
    # to the training size prints "N"; otherwise a range.
    Ms = [ci.get("M") for ci in cis]
    if not cis or all(m is None for m in Ms):
        n_out_label = None
    elif all(m == ci["N"] for m, ci in zip(Ms, cis)):
        n_out_label = "N"
    elif len(set(Ms)) == 1:
        n_out_label = str(Ms[0])
    else:
        n_out_label = "{}-{}".format(min(Ms), max(Ms))

    def _params(base):
        h = _append_q(list(base), q)
        if n_out_label is not None:
            h.append(mpatches.Patch(
                color="none", label=r"$N_{\mathrm{out}} = %s$" % n_out_label))
        return h

    legend_handles = _params(_hist_legend_handles(hist_bands, loss_label))

    saving = outdir is not None
    if saving:
        os.makedirs(outdir, exist_ok=True)
        base = os.path.join(outdir, "{}_{}".format(prefix, _stamp(stamp)))
        saved = []
    else:
        figs = []

    # per-sample-size histograms
    for r, ci in zip(results, cis):
        fig, ax = plt.subplots()
        _draw_ci_hist(ax, r["N"], r["F"], ci, hist_bands, loss_label)
        ax.set_ylabel("count")
        ax.legend(handles=legend_handles)
        fig.tight_layout()
        if saving:
            p = "{}_N{}.png".format(base, r["N"])
            fig.savefig(p, dpi=130); plt.close(fig); saved.append(p)
        else:
            figs.append((fig, ax))

    # combined small-multiples overview
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 3.4),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, r, ci in zip(axes, results, cis):
        _draw_ci_hist(ax, r["N"], r["F"], ci, hist_bands, loss_label)
    axes[0].set_ylabel("count")
    axes[-1].legend(handles=legend_handles, fontsize="small")
    fig.tight_layout()
    if saving:
        p = "{}_all.png".format(base)
        fig.savefig(p, dpi=130); plt.close(fig); saved.append(p)
    else:
        figs.append((fig, axes))

    # per-level: J_hat_N* vs N error bars, and half-width log-log scaling
    for level in levels:
        fig, ax = plt.subplots()
        _draw_ci_errorbar(ax, cis, level, value_label)
        if value_ylim is not None:
            ax.set_ylim(*value_ylim)
        extra = _params([])
        if extra:
            ax.legend(handles=extra)
        fig.tight_layout()
        if saving:
            p = "{}_ci{:.0f}.png".format(base, 100 * level)
            fig.savefig(p, dpi=130); plt.close(fig); saved.append(p)
        else:
            figs.append((fig, ax))

        if len(results) < 2:
            continue   # half-width vs N scaling needs at least two sample sizes
        fig, ax = plt.subplots()
        _draw_hw_loglog(ax, cis, level)
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=_params(handles))
        fig.tight_layout()
        if saving:
            p = "{}_ci{:.0f}_scaling.png".format(base, 100 * level)
            fig.savefig(p, dpi=130); plt.close(fig); saved.append(p)
        else:
            figs.append((fig, ax))

    if saving:
        print("[inference] wrote plug-in figures to {}_*".format(base))
        return saved
    return figs


# -- subsampling drawing helpers ---------------------------------------------

def _bmnq_handles(rec, q):
    """Invisible legend entries listing N, b, m (and q) for one sample size."""
    handles = [mpatches.Patch(color="none", label=r"$N = {}$".format(rec["N"])),
               mpatches.Patch(color="none", label=r"$b = {}$".format(rec["b"])),
               mpatches.Patch(color="none", label=r"$m = {}$".format(rec["m"]))]
    return _append_q(handles, q)


def _param_handles(results, key, label=None):
    """Legend patch for a per-N subsampling parameter (``key`` = "b"/"m").

    ``label`` (a LaTeX string) is printed verbatim as ``<key> = <label>`` if
    given.  Otherwise a single ``<key> = value`` when the value is constant across
    the sweep, else a single entry listing the per-N values on one line in
    sample-size order, ``<key> = v1, v2, ...`` (matching the N order on the
    x-axis)."""
    if label is not None:
        return [mpatches.Patch(color="none", label=r"${} = {}$".format(key, label))]
    vals = [r[key] for r in results]
    if len(set(vals)) == 1:
        return [mpatches.Patch(color="none",
                               label=r"${} = {}$".format(key, vals[0]))]
    joined = ", ".join(str(v) for v in vals)
    return [mpatches.Patch(color="none", label=r"${} = {}$".format(key, joined))]


def _bmq_handles(results, q, b_label=None, m_label=None):
    """Invisible legend entries listing b, m (and q) for the sweep plot; N is on
    the x-axis.  When b or m varies across the sweep, each sample size gets its
    own ``b_{N} = ...`` / ``m_{N} = ...`` entry; a constant value is shown once.
    ``b_label``/``m_label`` override with a verbatim LaTeX string."""
    handles = (_param_handles(results, "b", b_label)
               + _param_handles(results, "m", m_label))
    return _append_q(handles, q)


def _draw_subsampling_hist(ax, deltas, band, delta_label):
    ax.hist(deltas, bins="auto", density=True, color="0.8", edgecolor="0.4")
    ax.axvline(band["quantile_lo"], color=_BAND_COLOR, lw=1.5, ls="--",
               label=r"$\mathrm{quantile}_{N,b,m}(\beta/2)$")
    ax.axvline(band["quantile_hi"], color=_BAND_COLOR, lw=1.5, ls="--",
               label=r"$\mathrm{quantile}_{N,b,m}(1-\beta/2)$")
    ax.axvline(0.0, color="k", lw=1.0, ls=":")
    ax.set_xlabel(delta_label)
    ax.set_ylabel("density")
    ax.set_title("subsampling statistics")


def _draw_subsampling_ci_sweep(ax, cis, level, value_label):
    """J_hat_N* vs N across the sample sizes, with the (asymmetric) subsampling
    interval as an explicit [lo, hi] segment at each N -- the subsampling analogue
    of the plug-in error-bar figure.  Drawn as segments (not errorbar yerr)
    because the interval is asymmetric and need not bracket J_hat_N*, so signed
    offsets would be invalid as error bars."""
    Ns = [ci["N"] for ci in cis]
    Jhat = [ci["Jhat"] for ci in cis]
    for ci in cis:
        band = ci["levels"][level]
        N = ci["N"]
        ax.vlines(N, band["lo"], band["hi"], color="C1", lw=2)
        ax.scatter([N, N], [band["lo"], band["hi"]], marker="_", s=200,
                   linewidths=2, color="C1")
    ax.plot(Ns, Jhat, "o-", color="C0", zorder=3, label=r"$\widehat J_N^*$")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(value_label)
    ax.set_title("{} subsampling confidence interval".format(_pct(level)))
    ax.grid(True, which="both", alpha=0.3)


# -- public: subsampling figures ---------------------------------------------

def plot_subsampling(run_or_path, outdir=None, prefix="subsampling", stamp=None,
                     levels=None, band_level=0.95, q=None, labels=None,
                     value_ylim=None, b_label=None, m_label=None):
    """Render the subsampling figures from a run (dict) or a saved JSON ``path``.

    The run holds one record per sample size N (the subsampling analogue of the
    plug-in sweep).  Produces (i) a per-N density histogram of the Delta_r
    statistics with the empirical quantiles quantile(beta/2), quantile(1-beta/2)
    for ``band_level`` marked, and (ii) per-level confidence-interval plots
    across the sample sizes -- J_hat_N* vs N with the asymmetric subsampling
    interval drawn at each N, analogous to the plug-in ``_ci{level}`` figure.
    Legends list b, m, N and q (the control-mesh size).  ``q`` defaults to the
    value stored in the run.  ``value_ylim=(lo, hi)`` fixes the optimal-value (y)
    axis of the per-level ``_ci{level}`` interval plots -- pass the same value to
    ``plot_plugin`` so both algorithms' intervals share a y-axis for comparison.
    When ``outdir`` is given, saves PNGs named ``{prefix}_{stamp}_...`` and
    returns the list of paths; otherwise returns the list of ``(fig, ax)``.
    """
    configure_style()
    run = _as_subsampling_run(run_or_path)
    results = run["results"]
    levels = tuple(levels) if levels is not None else tuple(run["levels"])
    if band_level not in levels:
        levels = tuple(sorted(set(levels) | {band_level}))
    q = q if q is not None else run.get("q")
    _, value_label, delta_label = _labels(labels)
    # b/m legend text: explicit arg > expression stored in the run meta > value
    meta = run.get("meta") or {}
    b_label = b_label if b_label is not None else meta.get("b_expr")
    m_label = m_label if m_label is not None else meta.get("m_expr")

    cis = [subsampling_ci_from_deltas(r["deltas"], r["f_opt"], r["N"], levels)
           for r in results]

    saving = outdir is not None
    if saving:
        os.makedirs(outdir, exist_ok=True)
        base = os.path.join(outdir, "{}_{}".format(prefix, _stamp(stamp)))
        saved = []
    else:
        figs = []

    # per-N Delta_r histogram with the band_level quantiles marked
    for r, ci in zip(results, cis):
        band = ci["levels"][band_level]
        fig, ax = plt.subplots()
        _draw_subsampling_hist(ax, r["deltas"], band, delta_label)
        ci_handle = mpatches.Patch(color="none", label="{} CI: [{:.4g}, {:.4g}]"
                                   .format(_pct(band_level), band["lo"], band["hi"]))
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles + [ci_handle] + _bmnq_handles(r, q))
        fig.tight_layout()
        if saving:
            p = "{}_hist_N{}.png".format(base, r["N"])
            fig.savefig(p, dpi=130); plt.close(fig); saved.append(p)
        else:
            figs.append((fig, ax))

    # per-level subsampling CI across the sample sizes (plug-in-style sweep) and
    # the interval half-width vs N log-log scaling (like the plug-in _scaling)
    bmq = _bmq_handles(results, q, b_label=b_label, m_label=m_label)
    for level in levels:
        fig, ax = plt.subplots()
        _draw_subsampling_ci_sweep(ax, cis, level, value_label)
        if value_ylim is not None:
            ax.set_ylim(*value_ylim)
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles + bmq)
        fig.tight_layout()
        if saving:
            p = "{}_ci{:.0f}.png".format(base, 100 * level)
            fig.savefig(p, dpi=130); plt.close(fig); saved.append(p)
        else:
            figs.append((fig, ax))

        if len(results) < 2:
            continue   # half-width vs N scaling needs at least two sample sizes
        fig, ax = plt.subplots()
        _draw_hw_loglog(ax, cis, level)
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles + bmq)
        fig.tight_layout()
        if saving:
            p = "{}_ci{:.0f}_scaling.png".format(base, 100 * level)
            fig.savefig(p, dpi=130); plt.close(fig); saved.append(p)
        else:
            figs.append((fig, ax))

    if saving:
        print("[inference] wrote subsampling figures to {}_*".format(base))
        return saved
    return figs


# -- central-limit-theorem figures -------------------------------------------

def _as_clt_run(run_or_path):
    if isinstance(run_or_path, str):
        return load_clt_run(run_or_path)
    return run_or_path


def _draw_clt_hist(ax, stat):
    """Density histogram of the statistic for one N with a fitted normal curve."""
    stat = np.asarray(stat, dtype=float)
    ax.hist(stat, bins="auto", density=True, color="0.8", edgecolor="0.4")
    mu, sd = float(stat.mean()), float(stat.std())
    if sd > 0:
        xs = np.linspace(stat.min(), stat.max(), 200)
        ax.plot(xs, norm.pdf(xs, mu, sd), color="C0", lw=1.5)
    ax.axvline(0.0, color="k", lw=1.0, ls=":")
    ax.set_xlabel(r"$N^{1/2}(\widehat J_N^* - \widehat J_{\mathrm{ref}}^*)$")


def _clt_legend_handles(N, N_ref, R, q):
    handles = [mpatches.Patch(facecolor="0.8", edgecolor="0.4", label="empirical"),
               mlines.Line2D([], [], color="C0", lw=1.5, label="normal fit"),
               mlines.Line2D([], [], color="k", lw=1.0, ls=":", label="0"),
               mpatches.Patch(color="none", label=r"$N = {}$".format(N)),
               mpatches.Patch(color="none",
                              label=r"$N_{\mathrm{ref}} = %d$" % N_ref),
               mpatches.Patch(color="none", label=r"$R = {}$".format(R))]
    return _append_q(handles, q)


def plot_clt(run_or_path, outdir=None, prefix="clt", stamp=None, q=None,
             labels=None):
    """Render the limit-theorem histograms from a run (dict) or saved JSON ``path``.

    For each sample size N the statistic sqrt(N)*(J_hat_N* - J_hat_ref*) is
    recomputed from the stored raw replicate values (so the figures redraw with no
    re-solve) and shown as a density histogram with a fitted-normal overlay; as N
    grows the histogram should look increasingly Gaussian.  Legends list N,
    N_ref and q (the control-mesh size).  ``q`` defaults to the value stored in the
    run.  When ``outdir`` is given, saves ``{prefix}_{stamp}_clt_N{N}.png`` and a
    combined ``{prefix}_{stamp}_clt_all.png`` and returns the list of paths;
    otherwise returns the list of ``(fig, ax)``.
    """
    configure_style()
    run = _as_clt_run(run_or_path)
    results = run["results"]
    N_ref = run["N_ref"]
    f_ref = run["f_ref"]
    q = q if q is not None else run.get("q")
    stats = [clt_statistic(r["values"], r["N"], f_ref) for r in results]

    saving = outdir is not None
    if saving:
        os.makedirs(outdir, exist_ok=True)
        base = os.path.join(outdir, "{}_{}".format(prefix, _stamp(stamp)))
        saved = []
    else:
        figs = []

    # per-N histograms (each saved separately)
    for r, stat in zip(results, stats):
        fig, ax = plt.subplots()
        _draw_clt_hist(ax, stat)
        ax.set_ylabel("density")
        ax.legend(handles=_clt_legend_handles(r["N"], N_ref, stat.size, q))
        fig.tight_layout()
        if saving:
            p = "{}_clt_N{}.png".format(base, r["N"])
            fig.savefig(p, dpi=130); plt.close(fig); saved.append(p)
        else:
            figs.append((fig, ax))

    # combined small-multiples overview
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 3.4))
    axes = np.atleast_1d(axes)
    for ax, r, stat in zip(axes, results, stats):
        _draw_clt_hist(ax, stat)
        ax.legend(handles=_clt_legend_handles(r["N"], N_ref, stat.size, q),
                  fontsize="small")
    axes[0].set_ylabel("density")
    fig.suptitle(r"$N^{1/2}(\widehat J_N^* - \widehat J_{\mathrm{ref}}^*)$")
    fig.tight_layout()
    if saving:
        p = "{}_clt_all.png".format(base)
        fig.savefig(p, dpi=130); plt.close(fig); saved.append(p)
    else:
        figs.append((fig, axes))

    if saving:
        print("[inference] wrote CLT figures to {}_clt_*".format(base))
        return saved
    return figs
