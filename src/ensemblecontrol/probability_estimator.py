"""Clopper-Pearson lower confidence bound on a binomial success probability.

Implements the estimator p_hat_{N,delta}(L) of eq. (10.2.4) / Lemma 10.2.1: the
smallest success probability q whose Binomial(N, q) upper tail from L reaches
delta,

    p_hat_{N,delta}(L) = min{ q in [0,1] :
                              sum_{k=L}^{N} C(N,k) q^k (1-q)^{N-k} >= delta }.

The upper-tail sum is the regularized incomplete beta I_q(L, N-L+1) (increasing
in q), so the minimizer is the Clopper-Pearson lower bound beta.ppf(delta, L,
N-L+1), with the boundary values p_hat(0) = 0 and p_hat(N) = delta**(1/N).  By
Lemma 10.2.1, Prob{ p_hat(xi^N) > p } <= delta, i.e. p_hat is a valid (1-delta)
lower confidence bound on the true probability p.

The intended use is a coverage test: run N = R independent replications, let L be
the number in which a confidence interval covered the true value, and read
p_hat_{R,delta}(L) as a (1-delta) lower bound on the interval's true coverage
probability.  Note that ``delta`` here is the failure probability of *this* bound
(delta = 0.05 gives a 95%-confident lower bound), distinct from the confidence
level of the interval being tested.

This module depends only on numpy/scipy (no ensemblecontrol imports) and is
matplotlib-free.
"""

import numpy as np
from scipy.special import betainc
from scipy.stats import beta

__all__ = ["probability_lower_bound", "binomial_upper_tail"]


def _validate_counts(n, ell):
    """Validate integer n >= 1 and integer-valued ell (scalar or array) in [0, n]."""
    if int(n) != n or n < 1:
        raise ValueError("n must be an integer >= 1")
    arr = np.asarray(ell)
    if not np.all(np.equal(np.mod(arr, 1), 0)):
        raise ValueError("ell must be integer-valued")
    if np.any(arr < 0) or np.any(arr > n):
        raise ValueError("ell must lie in [0, n]")


def binomial_upper_tail(n, ell, q):
    """Upper-tail binomial sum sum_{k=ell}^n C(n,k) q^k (1-q)^{n-k}.

    Equal to the regularized incomplete beta I_q(ell, n-ell+1) for ell >= 1, and
    to 1 for ell = 0 (the whole mass).  ``ell`` and ``q`` may be scalars or
    arrays; scalar inputs return a Python ``float``.
    """
    _validate_counts(n, ell)
    qa = np.asarray(q, dtype=float)
    if np.any(qa < 0.0) or np.any(qa > 1.0):
        raise ValueError("q must lie in [0, 1]")
    ella = np.asarray(ell)
    # Feed a = 1 where ell == 0 so betainc stays finite, then overwrite with 1.0
    # (betainc(0, .) would be nan).
    a = np.where(ella == 0, 1, ella)
    tail = np.where(ella == 0, 1.0, betainc(a, n - ella + 1, qa))
    if np.ndim(ell) == 0 and np.ndim(q) == 0:
        return float(tail)
    return tail


def probability_lower_bound(n, ell, delta):
    """(1-delta) lower confidence bound p_hat_{n,delta}(ell) on a binomial prob.

    The smallest q with ``binomial_upper_tail(n, ell, q) >= delta``, i.e.
    ``scipy.stats.beta.ppf(delta, ell, n-ell+1)`` for ell >= 1, with the boundary
    values p_hat(0) = 0 and p_hat(n) = delta**(1/n).  ``ell`` may be a scalar or
    an array of counts; a scalar returns a Python ``float``.
    """
    _validate_counts(n, ell)
    d = float(delta)
    if not 0.0 < d < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    ella = np.asarray(ell)
    # a = 1 where ell == 0 keeps beta.ppf finite; the value is masked to 0 below.
    a = np.where(ella == 0, 1, ella)
    q = beta.ppf(d, a, n - ella + 1)
    q = np.where(ella == 0, 0.0, q)
    if np.ndim(ell) == 0:
        return float(q)
    return np.asarray(q, dtype=float)
