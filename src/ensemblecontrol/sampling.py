"""Reproducible scenario samplers for the SAA problems.

A sampler produces the ``(N, nparams)`` array of scenario parameters that
``SAAProblem`` consumes.  The base class handles reproducibility (a numpy
Generator seeded from ``seed``), the choice between i.i.d. Monte Carlo and Sobol
quasi-Monte Carlo unit draws, and ``spawn`` for independent replication streams.

The statistical procedures in :mod:`ensemblecontrol.inference` (plug-in and
subsampling confidence intervals) and the central-limit-theorem study all assume
**i.i.d.** scenarios, so use ``method="mc"`` for those.  Sobol QMC breaks the
i.i.d. assumption -- and subsets of a Sobol sequence are not Sobol, which also
disqualifies it for the subsampling algorithm -- so QMC is only appropriate for
deterministic control visualization.  ``spawn`` therefore refuses ``method="qmc"``.
"""

import copy

import numpy as np
from scipy.stats import truncnorm

__all__ = ["ScenarioSampler", "UniformSampler", "UniformRelativeSampler",
           "TruncatedNormalSampler"]


class ScenarioSampler(object):
    """Base class: RNG handling, unit draws, and independent replication streams.

    method : "mc" (i.i.d. Monte Carlo, default) or "qmc" (Sobol; N must be a
        power of two).
    seed   : seed/entropy for numpy's default_rng (None -> unpredictable).
    scramble : Sobol scrambling flag (QMC only).
    """

    def __init__(self, method="mc", seed=None, scramble=False):
        if method not in ("mc", "qmc"):
            raise ValueError("method must be 'mc' or 'qmc', got %r" % (method,))
        self.method = method
        self.seed = seed
        self.scramble = scramble
        self._rng = np.random.default_rng(seed)

    def _unit(self, N, d):
        """An (N, d) array of points in [0, 1)."""
        if self.method == "mc":
            return self._rng.random((N, d))
        from scipy.stats import qmc
        m = int(round(np.log2(N)))
        if 2 ** m != N:
            raise ValueError("QMC sampling requires N a power of two, got %d" % N)
        return qmc.Sobol(d=d, scramble=self.scramble).random_base2(m)

    def spawn(self, nreplications):
        """Return ``nreplications`` independent child samplers.

        Uses numpy's native stream splitting (``Generator.spawn`` /
        ``SeedSequence.spawn``) for statistically independent, well-separated
        streams -- no arithmetic seed offsets.  Only available for
        ``method="mc"``; QMC breaks i.i.d. replication.  Repeated calls on the
        same parent yield disjoint children, so calling ``spawn`` inside a loop
        gives a fresh independent batch each time.
        """
        if self.method != "mc":
            raise ValueError("spawn() requires method='mc'; QMC breaks i.i.d. "
                             "replication.")
        return [self._with_rng(child) for child in self._rng.spawn(nreplications)]

    def _with_rng(self, rng):
        clone = copy.copy(self)   # shallow: share config, replace only the RNG
        clone._rng = rng
        return clone

    def sample(self, N):
        """Return an (N, nparams) array of scenario parameters."""
        raise NotImplementedError

    def mean(self):
        """The expected parameter vector E[xi] (the nominal parameter).

        The nominal problem replaces the random parameter by its expectation, so
        the nominal solve uses ``sampler.mean()`` as its single scenario.
        """
        raise NotImplementedError


class UniformSampler(ScenarioSampler):
    """Independent uniform marginals: xi_j ~ U[lower_j, upper_j].

    lower/upper are scalars (one parameter) or equal-length sequences (one per
    parameter).  Reproduces the harmonic-oscillator draw of arXiv:2407.18182
    (Melnikov & Milz): the uncertain angular frequency k ~ U[0, 2*pi].
    """

    def __init__(self, lower, upper, method="mc", seed=None, scramble=False):
        super().__init__(method=method, seed=seed, scramble=scramble)
        self.lower = np.atleast_1d(np.asarray(lower, dtype=float))
        self.upper = np.atleast_1d(np.asarray(upper, dtype=float))

    def sample(self, N):
        u = self._unit(N, self.lower.size)
        return self.lower + u * (self.upper - self.lower)

    def mean(self):
        return 0.5 * (self.lower + self.upper)


class UniformRelativeSampler(ScenarioSampler):
    """Relative multiplicative perturbation: xi_j = (1 + radius * eta_j) * nominal_j.

    eta_j ~ U[-1, 1] (i.i.d. per parameter and scenario).  Columns listed in
    ``frozen`` are pinned to their nominal value (eta = 0), for parameters that
    carry no uncertainty.  Reproduces the fed-batch reactor scenario draw.
    """

    def __init__(self, nominal_param, radius, method="mc", seed=None,
                 frozen=(), scramble=False):
        super().__init__(method=method, seed=seed, scramble=scramble)
        self.nominal_param = np.asarray(nominal_param, dtype=float).ravel()
        self.radius = float(radius)
        self.frozen = tuple(frozen)

    def sample(self, N):
        d = self.nominal_param.size
        eta = 2.0 * self._unit(N, d) - 1.0
        if self.frozen:
            eta[:, list(self.frozen)] = 0.0
        return (1.0 + self.radius * eta) * self.nominal_param

    def mean(self):
        # E[(1 + radius*eta)*nominal] = nominal, since E[eta] = 0
        return self.nominal_param.copy()


class TruncatedNormalSampler(ScenarioSampler):
    """Truncated-normal marginals: xi_j ~ TruncatedNormal(mean_j, std_j) on
    [lower_j, upper_j].

    mean/std/lower/upper are scalars (one parameter) or equal-length sequences
    (independent per-parameter marginals).  Reproduces the constant-temperature
    batch-reactor draw, k20 ~ truncnorm(1000, 500, [500, 2000]).
    """

    def __init__(self, mean, std, lower, upper, method="mc", seed=None,
                 scramble=False):
        super().__init__(method=method, seed=seed, scramble=scramble)
        # loc/scale are the underlying-normal parameters; the truncated mean is
        # returned by mean() (it differs from loc under asymmetric truncation).
        self.loc = np.atleast_1d(np.asarray(mean, dtype=float))
        self.scale = np.atleast_1d(np.asarray(std, dtype=float))
        self.lower = np.atleast_1d(np.asarray(lower, dtype=float))
        self.upper = np.atleast_1d(np.asarray(upper, dtype=float))

    def _ab(self):
        return ((self.lower - self.loc) / self.scale,
                (self.upper - self.loc) / self.scale)

    def sample(self, N):
        a, b = self._ab()
        u = self._unit(N, self.loc.size)
        return truncnorm.ppf(u, a, b, loc=self.loc, scale=self.scale)

    def mean(self):
        a, b = self._ab()
        return truncnorm.mean(a, b, loc=self.loc, scale=self.scale)
