"""The distribution of the continue head."""

import jax.numpy as jnp

from dreamer.distributions import Bernoulli


def test_loss_with_a_soft_target_is_lowest_at_the_target_probability():
    # The continue head learns a soft target such as 0.997. The loss is the
    # cross-entropy, so the best prediction is that probability.
    logits = jnp.linspace(-10, 10, 2001)

    loss = -Bernoulli(logits).log_prob(0.997)

    best = Bernoulli(logits[jnp.argmin(loss)]).prob()
    assert jnp.abs(best - 0.997) < 1e-3


def test_hard_targets_prefer_confident_predictions():
    dist = Bernoulli(jnp.array([-5.0, 0.0, 5.0]))

    assert (jnp.diff(dist.log_prob(1.0)) > 0).all()
    assert (jnp.diff(dist.log_prob(0.0)) < 0).all()
