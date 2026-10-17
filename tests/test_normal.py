"""The policy distribution for continuous actions."""

import jax
import jax.numpy as jnp
import numpy as np

from dreamer.distributions import Normal, bounded_normal


def test_bounded_normal_keeps_the_mean_and_std_in_their_ranges():
    raw = jnp.array([-100.0, -1.0, 0.0, 1.0, 100.0])

    dist = bounded_normal(raw, raw, min_std=0.1, max_std=1.0)

    assert (jnp.abs(dist.mean) <= 1).all()
    assert (dist.std >= 0.1).all()
    assert (dist.std <= 1.0).all()


def test_bounded_normal_starts_with_a_large_std_for_exploration():
    dist = bounded_normal(jnp.zeros(6), jnp.zeros(6), min_std=0.1, max_std=1.0)

    np.testing.assert_allclose(dist.mean, 0.0)
    np.testing.assert_allclose(dist.std, 0.89, atol=0.01)


def test_samples_have_the_mean_and_std_of_the_distribution():
    dist = Normal(jnp.array([0.5, -0.3]), jnp.array([0.1, 0.8]))
    keys = jax.random.split(jax.random.key(0), 10_000)

    samples = jax.vmap(dist.sample)(keys)

    np.testing.assert_allclose(samples.mean(0), dist.mean, atol=0.02)
    np.testing.assert_allclose(samples.std(0), dist.std, atol=0.02)


def test_log_prob_is_highest_at_the_mean():
    dist = Normal(jnp.array(0.5), jnp.array(0.2))
    x = jnp.linspace(-1, 2, 301)

    assert x[jnp.argmax(dist.log_prob(x))] == 0.5


def test_entropy_increases_with_the_std():
    entropy = Normal(jnp.zeros(3), jnp.array([0.1, 0.5, 1.0])).entropy()

    assert entropy[0] < entropy[1] < entropy[2]
