"""The actor network and the slow critic."""

import jax
import numpy as np
from flax import nnx

from dreamer.actor_critic import Actor, update_slow_critic
from dreamer.world_model import TwoHotHead


def copy_of_weights(module):
    # `nnx.state` refers to the variables of the module and does not copy them.
    return jax.tree.map(np.array, nnx.state(module, nnx.Param))


def assert_trees_close(a, b, **tolerance):
    jax.tree.map(lambda x, y: np.testing.assert_allclose(x, y, **tolerance), a, b)


def test_actor_starts_with_the_same_wide_policy_in_every_state():
    actor = Actor(10, 3, units=16, layers=2, rngs=nnx.Rngs(0))
    features = 3 * jax.random.normal(jax.random.key(1), (100, 10))

    policy = actor(features)

    assert policy.mean.shape == (100, 3)
    np.testing.assert_allclose(policy.mean, 0.0, atol=0.05)
    np.testing.assert_allclose(policy.std, 0.89, atol=0.01)


def test_slow_critic_moves_toward_the_critic_by_the_rate():
    critic = TwoHotHead(4, units=8, layers=1, rngs=nnx.Rngs(0))
    slow_critic = TwoHotHead(4, units=8, layers=1, rngs=nnx.Rngs(1))
    before = copy_of_weights(slow_critic)
    target = copy_of_weights(critic)

    update_slow_critic(slow_critic, critic, rate=0.02)

    expected = jax.tree.map(lambda b, t: 0.98 * b + 0.02 * t, before, target)
    assert_trees_close(copy_of_weights(slow_critic), expected, rtol=1e-5)


def test_slow_critic_reaches_a_critic_that_does_not_change():
    critic = TwoHotHead(4, units=8, layers=1, rngs=nnx.Rngs(0))
    slow_critic = TwoHotHead(4, units=8, layers=1, rngs=nnx.Rngs(1))

    for _ in range(500):
        update_slow_critic(slow_critic, critic, rate=0.02)

    assert_trees_close(copy_of_weights(slow_critic), copy_of_weights(critic), atol=1e-4)
