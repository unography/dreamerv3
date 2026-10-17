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


def at_step(states, t):
    return jax.tree.map(lambda x: x[:, t], states)


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


T = 6  # Sequence length.


def random_sequence(key, is_first_at=()):
    k1, k2 = jax.random.split(key)
    prev_actions = jax.random.normal(k1, (B, T, A))
    tokens = jax.random.normal(k2, (B, T, E))
    is_first = jnp.zeros((B, T), bool).at[:, list(is_first_at)].set(True)
    return prev_actions, tokens, is_first


def test_observe_is_a_loop_of_obs_steps():
    rssm = make_rssm()
    prev_actions, tokens, is_first = random_sequence(jax.random.key(1), [0])
    key = jax.random.key(2)

    states, logits = rssm.observe(rssm.initial(B), prev_actions, tokens, is_first, key)

    state = rssm.initial(B)
    for t, step_key in enumerate(jax.random.split(key, T)):
        state, step_logits = rssm.obs_step(
            state, prev_actions[:, t], tokens[:, t], is_first[:, t], step_key
        )
        assert_states_close(at_step(states, t), state)
        np.testing.assert_allclose(logits[:, t], step_logits, atol=1e-6)


def test_is_first_inside_a_sequence_forgets_the_earlier_steps():
    rssm = make_rssm()
    prev_actions, tokens, is_first = random_sequence(jax.random.key(1), [0, 3])
    # A second sequence that differs only before the reset at step 3.
    other_tokens = tokens.at[:, :3].set(jax.random.normal(jax.random.key(2), (B, 3, E)))
    key = jax.random.key(3)

    states, _ = rssm.observe(rssm.initial(B), prev_actions, tokens, is_first, key)
    others, _ = rssm.observe(rssm.initial(B), prev_actions, other_tokens, is_first, key)

    assert not jnp.allclose(states.deter[:, 2], others.deter[:, 2])
    np.testing.assert_allclose(states.deter[:, 3:], others.deter[:, 3:])
    np.testing.assert_array_equal(states.stoch[:, 3:], others.stoch[:, 3:])


def test_deterministic_state_remembers_earlier_observations():
    rssm = make_rssm()
    prev_actions, tokens, is_first = random_sequence(jax.random.key(1), [0])

    def last_deter(first_token):
        changed = tokens.at[:, 0].set(first_token)
        states, _ = rssm.observe(
            rssm.initial(B), prev_actions, changed, is_first, jax.random.key(2)
        )
        return states.deter[:, -1].sum()

    grad = jax.grad(last_deter)(tokens[:, 0])

    assert (jnp.abs(grad) > 0).all()


def test_imagine_starts_from_the_start_state_and_follows_the_policy():
    rssm = make_rssm()
    start = random_state(rssm, jax.random.key(1))

    def forward(state, key):
        return jnp.ones((B, A))

    def backward(state, key):
        return -jnp.ones((B, A))

    states, actions = rssm.imagine(start, forward, horizon=4, key=jax.random.key(2))
    others, _ = rssm.imagine(start, backward, horizon=4, key=jax.random.key(2))

    assert states.deter.shape == (B, 5, 16)
    assert states.stoch.shape == (B, 5, 4, 3)
    assert_states_close(at_step(states, 0), start)
    np.testing.assert_array_equal(actions, jnp.ones((B, 4, A)))
    assert not jnp.allclose(states.deter[:, 1], others.deter[:, 1])
