"""The world model: encoder, decoder, heads, and its loss."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from dreamer.symlog import symlog
from dreamer.world_model import (
    Decoder,
    Encoder,
    TwoHotHead,
    WorldModel,
    continue_target,
    symlog_mse,
)

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


def test_two_hot_head_predicts_zero_at_the_start():
    head = TwoHotHead(10, units=8, layers=1, rngs=nnx.Rngs(0))
    features = jax.random.normal(jax.random.key(1), (4, 10))

    np.testing.assert_array_equal(head(features).mean(), 0.0)


def test_continue_target_is_the_discount_or_zero_at_terminal_steps():
    is_terminal = jnp.array([False, False, True])

    np.testing.assert_allclose(
        continue_target(is_terminal, horizon=100), [0.99, 0.99, 0.0]
    )


B, T, A = 4, 8, 2


def make_world_model():
    return WorldModel(
        OBS_SHAPES,
        A,
        deter=32,
        hidden=16,
        stoch=4,
        classes=4,
        blocks=2,
        units=16,
        rngs=nnx.Rngs(0),
    )


def fixed_batch():
    k1, k2, k3 = jax.random.split(jax.random.key(1), 3)
    data = random_obs(k1, (B, T))
    data["velocity"] *= 5
    data["reward"] = 3 * jax.random.uniform(k2, (B, T))
    data["is_first"] = jnp.zeros((B, T), bool).at[:, 0].set(True)
    data["is_terminal"] = jnp.zeros((B, T), bool).at[:, -1].set(True)
    prev_actions = jax.random.normal(k3, (B, T, A))
    return prev_actions, data


def test_world_model_gives_one_loss_value_for_each_step():
    model = make_world_model()
    prev_actions, data = fixed_batch()

    losses, states = model.loss(
        model.rssm.initial(B), prev_actions, data, jax.random.key(2)
    )

    assert set(losses) == {"dyn", "rep", "rec", "rew", "con"}
    assert all(loss.shape == (B, T) for loss in losses.values())
    assert states.deter.shape == (B, T, 32)


def test_world_model_learns_to_predict_a_fixed_batch():
    model = make_world_model()
    prev_actions, data = fixed_batch()

    @nnx.jit
    def train_step(model, key):
        def total_loss(model):
            losses, _ = model.loss(model.rssm.initial(B), prev_actions, data, key)
            return sum(loss.mean() for loss in losses.values()), losses

        grads, losses = nnx.grad(total_loss, has_aux=True)(model)
        params = nnx.state(model, nnx.Param)
        nnx.update(model, jax.tree.map(lambda p, g: p - 0.05 * g, params, grads))
        return {name: loss.mean() for name, loss in losses.items()}

    keys = jax.random.split(jax.random.key(2), 100)
    first = train_step(model, keys[0])
    for key in keys[1:]:
        last = train_step(model, key)

    for name in ("rec", "rew", "con"):
        assert last[name] < 0.6 * first[name], name
