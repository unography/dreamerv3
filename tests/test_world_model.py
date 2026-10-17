"""The world model: encoder, decoder, heads, and its loss."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from dreamer.symlog import symlog
from dreamer.world_model import Decoder, Encoder, symlog_mse

OBS_SHAPES = {"velocity": (3,), "height": (1,)}


def random_obs(key, batch_shape):
    k1, k2 = jax.random.split(key)
    return {
        "velocity": jax.random.normal(k1, (*batch_shape, 3)),
        "height": jax.random.normal(k2, (*batch_shape, 1)),
    }


def test_encoder_gives_one_token_for_each_step():
    encoder = Encoder(OBS_SHAPES, units=8, layers=2, rngs=nnx.Rngs(0))

    tokens = encoder(random_obs(jax.random.key(1), (4, 6)))

    assert tokens.shape == (4, 6, 8)


def test_encoder_does_not_depend_on_the_order_of_the_keys():
    encoder = Encoder(OBS_SHAPES, units=8, layers=2, rngs=nnx.Rngs(0))
    obs = random_obs(jax.random.key(1), (4,))

    reordered = {"height": obs["height"], "velocity": obs["velocity"]}

    np.testing.assert_array_equal(encoder(obs), encoder(reordered))


def test_decoder_predicts_each_observation_vector():
    decoder = Decoder(10, OBS_SHAPES, units=8, layers=2, rngs=nnx.Rngs(0))

    prediction = decoder(jnp.ones((4, 6, 10)))

    assert prediction["velocity"].shape == (4, 6, 3)
    assert prediction["height"].shape == (4, 6, 1)


def test_symlog_mse_is_zero_for_a_prediction_of_the_symlog_target():
    target = jnp.array([[-500.0, 0.0, 3.0]])

    np.testing.assert_allclose(symlog_mse(symlog(target), target), [0.0])


def test_symlog_mse_keeps_large_errors_small():
    # In symlog space, a prediction of 1000 for a target of 2000 is a small
    # error. The plain squared error is 1e6.
    loss = symlog_mse(symlog(jnp.array([1000.0])), jnp.array([2000.0]))

    assert loss < 0.5
