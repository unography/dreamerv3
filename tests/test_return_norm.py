"""Return normalization: the scale of the advantages for the actor."""

import jax
import jax.numpy as jnp
import numpy as np

from dreamer.actor_critic import ReturnNorm


def scale_after_updates(returns_fn, num_updates=1000):
    norm = ReturnNorm()
    for key in jax.random.split(jax.random.key(0), num_updates):
        norm.update(returns_fn(key))
    return norm.scale()


def test_scale_is_the_range_of_large_returns():
    # The range from the 5th to the 95th percentile of U(0, 1000) is 900.
    scale = scale_after_updates(lambda key: jax.random.uniform(key, (1024,)) * 1000)

    np.testing.assert_allclose(scale, 900, rtol=0.02)


def test_scale_does_not_go_below_one_for_small_returns():
    scale = scale_after_updates(lambda key: jax.random.uniform(key, (1024,)) * 0.1)

    assert scale == 1.0


def test_rare_outliers_do_not_change_the_scale():
    def returns(key):
        returns = jax.random.uniform(key, (1000,)) * 100
        return returns.at[:10].set(1e6)  # 1% of the returns are outliers.

    np.testing.assert_allclose(scale_after_updates(returns), 90, rtol=0.05)


def test_one_update_moves_the_statistics_by_the_rate():
    norm = ReturnNorm(rate=0.01)

    norm.update(jnp.linspace(0, 1000, 1001))

    np.testing.assert_allclose(norm.scale(), 0.01 * 900, rtol=1e-5)
