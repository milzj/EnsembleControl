import numpy as np
import pytest

from ensemblecontrol import (ScenarioSampler, UniformSampler,
                             UniformRelativeSampler, TruncatedNormalSampler)


def test_uniform_sampler_support_and_shape():
    # k ~ U[0, 2*pi], as in the harmonic-oscillator example
    sampler = UniformSampler(0.0, 2 * np.pi, seed=3)
    X = sampler.sample(64)
    assert X.shape == (64, 1)
    assert X.min() >= 0.0
    assert X.max() <= 2 * np.pi


def test_uniform_sampler_vector_marginals():
    sampler = UniformSampler([0.0, -1.0], [1.0, 1.0], seed=4)
    X = sampler.sample(32)
    assert X.shape == (32, 2)
    assert (X[:, 0] >= 0).all() and (X[:, 0] <= 1).all()
    assert (X[:, 1] >= -1).all() and (X[:, 1] <= 1).all()


def test_uniform_relative_shape_bounds_and_frozen():
    nominal = [2.0, 4.0, 20.0]
    sampler = UniformRelativeSampler(nominal, radius=0.2, seed=0, frozen=(2,))
    X = sampler.sample(16)
    assert X.shape == (16, 3)
    # frozen column pinned to its nominal value
    assert np.allclose(X[:, 2], 20.0)
    # perturbed columns stay within +/- radius of nominal
    for j, nom in enumerate(nominal[:2]):
        assert X[:, j].min() >= (1 - 0.2) * nom - 1e-9
        assert X[:, j].max() <= (1 + 0.2) * nom + 1e-9


def test_seed_reproducible():
    nominal = [1.0, 2.0]
    a = UniformRelativeSampler(nominal, 0.1, seed=7).sample(8)
    b = UniformRelativeSampler(nominal, 0.1, seed=7).sample(8)
    assert np.array_equal(a, b)


def test_spawn_independent_and_reproducible():
    nominal = [1.0, 2.0]
    child1a, child1b = UniformRelativeSampler(nominal, 0.1, seed=0).spawn(2)
    child2a, child2b = UniformRelativeSampler(nominal, 0.1, seed=0).spawn(2)
    # same parent seed -> identical spawned streams (reproducible)
    assert np.array_equal(child1a.sample(4), child2a.sample(4))
    assert np.array_equal(child1b.sample(4), child2b.sample(4))
    # distinct children -> independent (different) draws
    assert not np.array_equal(child1a.sample(4), child1b.sample(4))


def test_spawn_rejects_qmc():
    sampler = UniformRelativeSampler([1.0], 0.1, method="qmc", seed=0)
    with pytest.raises(ValueError):
        sampler.spawn(2)


def test_qmc_requires_power_of_two():
    sampler = UniformRelativeSampler([1.0, 2.0], 0.1, method="qmc")
    sampler.sample(8)                      # power of two is fine
    with pytest.raises(ValueError):
        sampler.sample(6)                  # not a power of two


def test_truncated_normal_support_and_shape():
    sampler = TruncatedNormalSampler(1000.0, 500.0, 500.0, 2000.0, seed=1)
    X = sampler.sample(64)
    assert X.shape == (64, 1)
    assert X.min() >= 500.0 - 1e-6
    assert X.max() <= 2000.0 + 1e-6


def test_truncated_normal_vector_marginals():
    sampler = TruncatedNormalSampler([0.5, 1.0], [0.1, 0.2], [0.0, 0.0],
                                     [1.0, 2.0], seed=2)
    X = sampler.sample(32)
    assert X.shape == (32, 2)
    assert (X[:, 0] >= 0).all() and (X[:, 0] <= 1).all()
    assert (X[:, 1] >= 0).all() and (X[:, 1] <= 2).all()


def test_invalid_method():
    with pytest.raises(ValueError):
        ScenarioSampler(method="not-a-method")


def test_sampler_mean_is_expected_parameter():
    # E[xi] -- the nominal parameter (the nominal problem uses sampler.mean())
    assert UniformSampler(0.0, 2 * np.pi).mean()[0] == pytest.approx(np.pi)
    assert np.allclose(
        UniformRelativeSampler([2.0, 4.0], 0.2).mean(), [2.0, 4.0])
    # symmetric truncation -> truncated mean equals the center
    assert TruncatedNormalSampler(1.0, 0.5, 0.0, 2.0).mean()[0] == pytest.approx(1.0)
    # empirical mean of a large i.i.d. draw is close to mean()
    s = UniformSampler(0.0, 2 * np.pi, seed=0)
    assert s.sample(4096).mean(axis=0)[0] == pytest.approx(s.mean()[0], abs=0.2)
