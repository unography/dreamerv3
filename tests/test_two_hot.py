"""Two-hot regression: how DreamerV3 predicts rewards and values of any size."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dreamer.distributions import TwoHot, symexp_bins, two_hot
from dreamer.symlog import symexp


def test_bins_are_symmetric_around_zero():
    bins = symexp_bins(255)

    assert bins[127] == 0
    np.testing.assert_array_equal(bins, -bins[::-1])
    np.testing.assert_allclose(bins[-1], symexp(20.0))


def test_two_hot_interpolates_between_the_two_nearest_bins():
    bins = jnp.array([0.0, 1.0, 2.0, 4.0])

    np.testing.assert_allclose(two_hot(jnp.array(1.5), bins), [0, 0.5, 0.5, 0])
    np.testing.assert_allclose(two_hot(jnp.array(3.0), bins), [0, 0, 0.5, 0.5])
    np.testing.assert_allclose(two_hot(jnp.array(1.0), bins), [0, 1, 0, 0])


def test_two_hot_puts_values_outside_the_bins_on_the_edge_bin():
    bins = jnp.array([0.0, 1.0, 2.0, 4.0])

    np.testing.assert_allclose(two_hot(jnp.array(-5.0), bins), [1, 0, 0, 0])
    np.testing.assert_allclose(two_hot(jnp.array(9.0), bins), [0, 0, 0, 1])


@pytest.mark.parametrize("target", [-1e5, -3.3, 0.0, 1e-3, 0.5, 42.0, 1e6])
def test_two_hot_encoding_keeps_the_value_at_any_scale(target):
    bins = symexp_bins()
    logits = jnp.log(two_hot(jnp.array(target), bins))

    np.testing.assert_allclose(TwoHot(logits, bins).mean(), target, rtol=1e-4)


def test_uniform_logits_predict_exactly_zero():
    # The reward and value heads start with zero weights, so their logits are
    # uniform. They then predict exactly zero at the start of training.
    dist = TwoHot(jnp.zeros(255), symexp_bins())

    assert dist.mean() == 0.0


@pytest.mark.parametrize("target", [1.0, 1e3, 1e6])
def test_loss_gradient_has_the_same_bound_for_any_target(target):
    bins = symexp_bins()

    def loss(logits):
        return TwoHot(logits, bins).loss(jnp.array(target))

    grad = jax.grad(loss)(jnp.zeros(255))

    assert jnp.abs(grad).sum() <= 2.0
