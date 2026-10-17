"""The recurrent state-space model: one step, sequences, and its losses."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from dreamer.rssm import RSSM, State

B, E, A = 3, 5, 2  # Batch size, token size, action size.


def make_rssm():
    return RSSM(
        E, A, deter=16, hidden=8, stoch=4, classes=3, blocks=2, rngs=nnx.Rngs(0)
    )


def random_state(rssm, key):
    k1, k2 = jax.random.split(key)
    deter = jax.random.uniform(k1, (B, rssm.deter_size), minval=-1, maxval=1)
    index = jax.random.randint(k2, (B, rssm.stoch_size), 0, rssm.classes)
    return State(deter, jax.nn.one_hot(index, rssm.classes))


def assert_states_close(a, b):
    jax.tree.map(lambda x, y: np.testing.assert_allclose(x, y, atol=1e-6), a, b)


def test_obs_step_gives_a_one_hot_stochastic_state():
    rssm = make_rssm()
    token = jax.random.normal(jax.random.key(1), (B, E))
    no_reset = jnp.zeros(B, bool)

    state, _ = rssm.obs_step(
        rssm.initial(B), jnp.zeros((B, A)), token, no_reset, jax.random.key(2)
    )

    assert state.stoch.shape == (B, 4, 3)
    np.testing.assert_array_equal(state.stoch.sum(-1), 1.0)


def test_is_first_starts_the_step_from_the_initial_state():
    rssm = make_rssm()
    old_state = random_state(rssm, jax.random.key(1))
    action = jax.random.normal(jax.random.key(2), (B, A))
    token = jax.random.normal(jax.random.key(3), (B, E))
    key = jax.random.key(4)

    reset = rssm.obs_step(old_state, action, token, jnp.ones(B, bool), key)
    fresh = rssm.obs_step(
        rssm.initial(B), jnp.zeros((B, A)), token, jnp.zeros(B, bool), key
    )

    assert_states_close(reset, fresh)


def test_only_the_stochastic_state_sees_the_current_observation():
    # h_t comes from the last state and action. The token e_t only changes the
    # posterior of z_t.
    rssm = make_rssm()
    state = random_state(rssm, jax.random.key(1))
    action = jax.random.normal(jax.random.key(2), (B, A))
    token_a, token_b = jax.random.normal(jax.random.key(3), (2, B, E))
    no_reset = jnp.zeros(B, bool)
    key = jax.random.key(4)

    state_a, logits_a = rssm.obs_step(state, action, token_a, no_reset, key)
    state_b, logits_b = rssm.obs_step(state, action, token_b, no_reset, key)

    np.testing.assert_array_equal(state_a.deter, state_b.deter)
    assert not jnp.allclose(logits_a, logits_b)


def test_deterministic_state_stays_in_the_range_of_tanh():
    rssm = make_rssm()
    state = rssm.initial(B)
    for key in jax.random.split(jax.random.key(1), 50):
        action = 10 * jax.random.normal(key, (B, A))
        state, _ = rssm.img_step(state, action, key)

        assert (jnp.abs(state.deter) <= 1).all()


def test_actions_outside_the_range_have_the_same_effect_as_the_bound():
    rssm = make_rssm()
    state = random_state(rssm, jax.random.key(1))

    np.testing.assert_allclose(
        rssm.recurrent(state, jnp.full((B, A), 5.0)),
        rssm.recurrent(state, jnp.full((B, A), 1.0)),
    )
