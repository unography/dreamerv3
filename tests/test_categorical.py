"""The categorical latents of the world model: unimix and straight-through."""

import jax
import jax.numpy as jnp
import numpy as np

from dreamer.distributions import OneHotCategorical

CONFIDENT = jnp.array([50.0, 0.0, 0.0, 0.0])


def test_unimix_gives_each_class_a_minimum_probability():
    dist = OneHotCategorical(CONFIDENT, unimix=0.01)

    np.testing.assert_allclose(dist.probs, [0.9925, 0.0025, 0.0025, 0.0025])


def test_unimix_limits_the_kl_divergence():
    a = OneHotCategorical(CONFIDENT, unimix=0.01)
    b = OneHotCategorical(CONFIDENT[::-1], unimix=0.01)
    a_pure = OneHotCategorical(CONFIDENT)
    b_pure = OneHotCategorical(CONFIDENT[::-1])

    assert a.kl(b) < 6.0
    assert a_pure.kl(b_pure) > 49.0


def test_kl_divergence_is_zero_only_for_equal_distributions():
    a = OneHotCategorical(jnp.array([1.0, 2.0, 3.0]))
    b = OneHotCategorical(jnp.array([3.0, 2.0, 1.0]))

    np.testing.assert_allclose(a.kl(a), 0.0, atol=1e-6)
    assert a.kl(b) > 0.1


def test_samples_are_one_hot_vectors_with_the_right_frequencies():
    dist = OneHotCategorical(jnp.log(jnp.array([0.1, 0.2, 0.7])))
    keys = jax.random.split(jax.random.key(0), 10_000)

    samples = jax.vmap(dist.sample)(keys)

    assert set(np.unique(samples).tolist()) == {0.0, 1.0}
    np.testing.assert_array_equal(samples.sum(-1), 1.0)
    np.testing.assert_allclose(samples.mean(0), [0.1, 0.2, 0.7], atol=0.02)


def test_sample_has_the_gradient_of_the_probabilities():
    logits = jnp.array([0.5, -1.0, 2.0])
    weights = jnp.array([1.0, 2.0, 3.0])

    def of_sample(logits):
        sample = OneHotCategorical(logits).sample(jax.random.key(0))
        return (weights * sample).sum()

    def of_probs(logits):
        return (weights * OneHotCategorical(logits).probs).sum()

    np.testing.assert_allclose(jax.grad(of_sample)(logits), jax.grad(of_probs)(logits))
