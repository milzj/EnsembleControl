"""Clopper-Pearson lower confidence bound on a binomial success probability.

Implements the estimator of Nemirovski, section 10.2.1 ("Simulation-Based
Justification"), https://www2.isye.gatech.edu/~nemirovs/FullBookDec11.pdf:

    p_hat_{N,delta}(L) = min{ q in [0,1] :
                              sum_{k=L}^{N} C(N,k) q^k (1-q)^{N-k} >= delta }   (10.2.4)

the smallest success probability q whose Binomial(N, q) upper tail from L reaches
delta.  The upper-tail sum is increasing in q, so the minimizer is the
Clopper-Pearson lower bound beta.ppf(delta, L, N-L+1), with boundary values
p_hat(0) = 0 and p_hat(N) = delta**(1/N).  By Lemma 10.2.1, Prob{ p_hat(xi^N) > p }
<= delta, i.e. p_hat is a valid (1-delta) lower confidence bound on the true
probability p.

The intended use is a coverage test: run N = R independent replications, let L be
the number in which a confidence interval covered the true value, and read
p_hat_{R,delta}(L) as a (1-delta) lower bound on the interval's true coverage
probability.  Here ``delta`` is the failure probability of *this* bound (delta =
0.05 gives a 95%-confident lower bound), distinct from the confidence level of the
interval being tested.

This module depends only on numpy/scipy (no ensemblecontrol imports) and is
matplotlib-free.
"""

import numpy as np
from scipy.stats import beta, binom

__all__ = ["probability_lower_bound", "binomial_upper_tail"]


def _validate(n, ell, delta):
    """Check integer n >= 1, integer-valued ell (scalar or array) in [0, n], and
    0 < delta < 1."""
    a = np.asarray(ell)
    if int(n) != n or n < 1:
        raise ValueError("n must be an integer >= 1")
    if np.any(np.mod(a, 1) != 0) or np.any(a < 0) or np.any(a > n):
        raise ValueError("ell must be an integer in [0, n]")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")


def binomial_upper_tail(n, ell, q):
    """Upper-tail binomial sum P(Binomial(n, q) >= ell) = sum_{k=ell}^n C(n,k) q^k
    (1-q)^{n-k}.

    A thin wrapper around ``scipy.stats.binom.sf``: ``sf(ell-1, n, q)`` is
    P(X > ell-1) = P(X >= ell), which handles ell = 0 (-> 1) and ell = n (-> q**n)
    for free.  ``ell`` and ``q`` may be scalars or arrays; a scalar returns a
    Python ``float``.
    """
    tail = binom.sf(np.asarray(ell) - 1, n, q)
    return float(tail) if np.ndim(tail) == 0 else tail


def probability_lower_bound(n, ell, delta=1e-6):
    """(1-delta) lower confidence bound p_hat_{n,delta}(ell) on a binomial prob.

    The smallest q with ``binomial_upper_tail(n, ell, q) >= delta``, i.e.
    ``scipy.stats.beta.ppf(delta, ell, n-ell+1)`` for ell >= 1, with the boundary
    values p_hat(0) = 0 and p_hat(n) = delta**(1/n).  ``ell`` may be a scalar or an
    array of counts; a scalar returns a Python ``float``.  ``delta`` defaults to
    1e-6 (a (1 - 1e-6) lower bound).
    """
    _validate(n, ell, delta)
    ell = np.asarray(ell)
    # beta.ppf(delta, 0, .) is nan, so feed a = 1 where ell == 0 and overwrite it.
    q = np.where(ell == 0, 0.0,
                 beta.ppf(delta, np.where(ell == 0, 1, ell), n - ell + 1))
    return float(q) if np.ndim(q) == 0 else np.asarray(q, dtype=float)
