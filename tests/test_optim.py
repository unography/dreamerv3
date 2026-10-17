"""The optimizer: adaptive gradient clipping, LaProp, and the warmup."""

import jax
import jax.numpy as jnp
import numpy as np

from dreamer.optim import clip_by_agc, make_optimizer


def apply(transform, grads, params, steps=1):
    """Gives the updates of the last of `steps` updates with the same gradient."""
    state = transform.init(params)
    for _ in range(steps):
        updates, state = transform.update(grads, state, params)
    return updates


def test_agc_limits_the_gradient_norm_to_a_fraction_of_the_weight_norm():
    params = {"w": jnp.full((10, 10), 1.0)}  # The weight norm is 10.
    grads = {"w": jnp.full((10, 10), 10.0)}  # The gradient norm is 100.

    updates = apply(clip_by_agc(0.3), grads, params)

    np.testing.assert_allclose(jnp.linalg.norm(updates["w"]), 3.0, rtol=1e-5)


def test_agc_does_not_change_small_gradients():
    params = {"w": jnp.full((10, 10), 1.0)}
    grads = {"w": jnp.full((10, 10), 0.1)}  # The gradient norm is 1.

    updates = apply(clip_by_agc(0.3), grads, params)

    np.testing.assert_array_equal(updates["w"], grads["w"])


def test_agc_clips_each_weight_array_on_its_own():
    params = {"small": jnp.ones(4), "large": jnp.ones(4) * 100}
    grads = {"small": jnp.ones(4) * 10, "large": jnp.ones(4) * 10}

    updates = apply(clip_by_agc(0.3), grads, params)

    assert jnp.linalg.norm(updates["small"]) < jnp.linalg.norm(grads["small"])
    np.testing.assert_array_equal(updates["large"], grads["large"])


def test_update_size_does_not_depend_on_the_gradient_scale():
    optimizer = make_optimizer(lr=1e-3, agc=1e9, warmup=0)
    params = {"w": jnp.ones(5)}
    grads = {"w": jax.random.normal(jax.random.key(0), (5,))}

    small = apply(optimizer, grads, params, steps=10)
    large = apply(optimizer, {"w": 1000 * grads["w"]}, params, steps=10)

    assert (small["w"] != 0).all()
    np.testing.assert_allclose(small["w"], large["w"], rtol=1e-5)


def test_each_element_moves_by_the_learning_rate_against_its_gradient_sign():
    optimizer = make_optimizer(lr=1e-3, agc=1e9, warmup=0)
    params = {"w": jnp.ones(3)}
    grads = {"w": jnp.array([1e-4, -2.0, 30.0])}

    updates = apply(optimizer, grads, params, steps=5)

    np.testing.assert_allclose(updates["w"], [-1e-3, 1e-3, -1e-3], rtol=1e-5)


def test_learning_rate_warms_up_linearly():
    optimizer = make_optimizer(lr=1e-3, agc=1e9, warmup=100)
    params = {"w": jnp.ones(1)}
    grads = {"w": jnp.ones(1)}

    sizes = [-apply(optimizer, grads, params, steps)["w"][0] for steps in (1, 51, 200)]

    np.testing.assert_allclose(sizes, [0.0, 0.5e-3, 1e-3], rtol=1e-5)
