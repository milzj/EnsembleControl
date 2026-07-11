from math import comb

import numpy as np
import pytest
from scipy.stats import beta, binom

import ensemblecontrol
from ensemblecontrol import binomial_upper_tail, probability_lower_bound


def test_lower_bound_edge_cases():
    # p_hat(n, 0, delta) = 0 (the tail is 1 >= delta for every q).
    for delta in (0.05, 0.5):
        assert probability_lower_bound(10, 0, delta) == 0.0
    # p_hat(n, n, delta) = delta**(1/n): tail = q**n >= delta.
    assert probability_lower_bound(10, 10, 0.05) == pytest.approx(0.05 ** 0.1)


def test_lower_bound_matches_beta_ppf():
    n, delta = 50, 0.05
    got = probability_lower_bound(n, np.arange(1, n + 1), delta)
    want = beta.ppf(delta, np.arange(1, n + 1), n - np.arange(1, n + 1) + 1)
    assert np.allclose(got, want)


def test_vectorized_over_ell():
    n, delta = 40, 0.05
    ell = np.array([0, 1, 7, 40])
    got = probability_lower_bound(n, ell, delta)
    scalar = np.array([probability_lower_bound(n, int(e), delta) for e in ell])
    assert got.shape == ell.shape
    assert np.allclose(got, scalar)
    assert got[0] == 0.0            # the ell == 0 mask survives the array path


def test_upper_tail_matches_binom_sf():
    n, q = 20, 0.3
    ell = np.arange(1, n + 1)
    got = binomial_upper_tail(n, ell, q)
    assert np.allclose(got, binom.sf(ell - 1, n, q))
    assert binomial_upper_tail(n, 0, q) == 1.0


def test_upper_tail_bruteforce_definition():
    n, q = 6, 0.4
    for ell in range(0, n + 1):
        brute = sum(comb(n, k) * q ** k * (1 - q) ** (n - k) for k in range(ell, n + 1))
        assert binomial_upper_tail(n, ell, q) == pytest.approx(brute)


def test_upper_tail_q_edges():
    n = 12
    for ell in range(1, n + 1):
        assert binomial_upper_tail(n, ell, 0.0) == 0.0
        assert binomial_upper_tail(n, ell, 1.0) == 1.0
    assert binomial_upper_tail(n, 0, 0.0) == 1.0


def test_roundtrip_tail_of_lower_bound():
    # For interior ell the tail at the lower bound equals delta exactly.
    n, delta = 30, 0.1
    for ell in range(1, n + 1):
        q = probability_lower_bound(n, ell, delta)
        assert binomial_upper_tail(n, ell, q) == pytest.approx(delta)


def test_monotone_in_ell():
    vals = probability_lower_bound(40, np.arange(0, 41), 0.05)
    assert np.all(np.diff(vals) >= -1e-12)


def test_monotone_in_delta():
    deltas = np.linspace(0.01, 0.99, 50)
    vals = np.array([probability_lower_bound(40, 10, d) for d in deltas])
    assert np.all(np.diff(vals) > 0)


@pytest.mark.parametrize("n,ell,delta", [
    (50, -1, 0.05),      # ell below 0
    (50, 51, 0.05),      # ell above n
    (50, 2.5, 0.05),     # non-integer ell
    (0, 0, 0.05),        # n < 1
    (50, 3, 0.0),        # delta <= 0
    (50, 3, 1.0),        # delta >= 1
])
def test_lower_bound_validation(n, ell, delta):
    with pytest.raises(ValueError):
        probability_lower_bound(n, ell, delta)


def test_lemma_10_2_1_guarantee_montecarlo():
    # Simulate xi ~ Bernoulli(p_true) and check Prob{p_hat > p_true} <= delta.
    rng = np.random.default_rng(0)
    p_true, n, delta, trials = 0.7, 50, 0.1, 20000
    ell = rng.binomial(n, p_true, trials)
    phat = probability_lower_bound(n, ell, delta)
    exceed = float(np.mean(phat > p_true))
    assert exceed <= delta + 0.02      # Lemma 10.2.1 (small slack for MC noise)
    assert exceed > 0.03               # and the bound is not vacuously loose


def test_default_delta_is_1e_6():
    assert probability_lower_bound(500, 480) == probability_lower_bound(500, 480, 1e-6)


def test_exports_available():
    assert ensemblecontrol.probability_lower_bound is probability_lower_bound
    assert ensemblecontrol.binomial_upper_tail is binomial_upper_tail
