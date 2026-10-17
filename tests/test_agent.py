"""The training step and the policy of the agent."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from dreamer.agent import Agent
from dreamer.config import Config

B, T, A = 4, 8, 2  # Batch size, sequence length, action size.
OBS_SHAPES = {"position": (3,), "velocity": (2,)}
SMALL = Config(
    deter=16, hidden=8, stoch=4, classes=3, blocks=2, units=8, lr=1e-3, warmup=0
)


def make_agent(config=SMALL):
    return Agent(OBS_SHAPES, A, config, rngs=nnx.Rngs(0))


def replay_batch():
    """A batch of sequences in the format of `Replay.sample`."""
    rng = np.random.default_rng(0)
    stoch = rng.integers(0, SMALL.classes, size=(B, T, SMALL.stoch))
    batch = {
        "position": rng.normal(size=(B, T, 3)).astype(np.float32),
        "velocity": rng.normal(size=(B, T, 2)).astype(np.float32),
        "action": rng.uniform(-1, 1, size=(B, T, A)).astype(np.float32),
        "reward": rng.uniform(size=(B, T)).astype(np.float32),
        "is_first": np.zeros((B, T), bool),
        "is_last": np.zeros((B, T), bool),
        "is_terminal": np.zeros((B, T), bool),
        # The RSSM states that the policy or the training stored.
        "deter": rng.uniform(-1, 1, size=(B, T, SMALL.deter)).astype(np.float32),
        "stoch": np.eye(SMALL.classes, dtype=np.float32)[stoch],
    }
    batch["is_first"][:, 0] = True
    return batch


def copy_of_weights(module):
    return jax.tree.map(np.array, nnx.state(module, nnx.Param))


def test_train_step_gives_finite_metrics():
    metrics, _ = make_agent().train(replay_batch(), jax.random.key(1))

    assert {"loss/rec", "loss/policy", "loss/value", "grad_norm"} <= set(metrics)
    assert all(jnp.isfinite(value) for value in metrics.values())


def test_training_changes_all_trained_weights():
    # The reward head and the critic start with zero output weights. Thus in
    # the first step, no gradient reaches their hidden layers, and the
    # advantages of the actor are zero. After a few steps, all weights learn.
    agent = make_agent()
    before = copy_of_weights(agent)

    for key in jax.random.split(jax.random.key(1), 3):
        agent.train(replay_batch(), key)

    after = copy_of_weights(agent)
    changed = jax.tree.map(lambda b, a: bool((b != a).any()), before, after)
    for name in ("world_model", "actor", "critic"):
        assert all(jax.tree.leaves(changed[name])), name


def test_slow_critic_only_moves_toward_the_critic():
    agent = make_agent()
    slow_before = copy_of_weights(agent.slow_critic)

    agent.train(replay_batch(), jax.random.key(1))

    expected = jax.tree.map(
        lambda slow, fast: 0.98 * slow + 0.02 * fast,
        slow_before,
        copy_of_weights(agent.critic),
    )
    jax.tree.map(
        lambda x, y: np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-7),
        copy_of_weights(agent.slow_critic),
        expected,
    )


def test_train_steps_on_one_batch_lower_the_world_model_losses():
    agent = make_agent()
    batch = replay_batch()

    keys = jax.random.split(jax.random.key(1), 30)
    first, _ = agent.train(batch, keys[0])
    for key in keys[1:]:
        last, _ = agent.train(batch, key)

    assert last["loss/rec"] < 0.8 * first["loss/rec"]
    assert last["loss/con"] < 0.8 * first["loss/con"]


def test_train_gives_the_states_of_the_trained_steps():
    _, states = make_agent().train(replay_batch(), jax.random.key(1))

    # With the replay context, the first step of each sequence only gives the
    # start state, so the training uses the other T - 1 steps.
    assert states.deter.shape == (B, T - 1, SMALL.deter)
    assert states.stoch.shape == (B, T - 1, SMALL.stoch, SMALL.classes)


def train_metrics(batch, config=SMALL):
    metrics, _ = make_agent(config).train(batch, jax.random.key(1))
    return metrics


def test_replay_context_uses_only_the_state_of_the_first_step():
    batch = replay_batch()
    other_obs = {**batch, "position": batch["position"].copy()}
    other_obs["position"][:, 0] += 10.0
    other_state = {**batch, "deter": batch["deter"].copy()}
    other_state["deter"][:, 0] *= -1

    metrics = train_metrics(batch)

    assert train_metrics(other_obs)["loss/rec"] == metrics["loss/rec"]
    assert train_metrics(other_state)["loss/rec"] != metrics["loss/rec"]


def test_without_replay_context_sequences_start_from_the_initial_state():
    config = dataclasses.replace(SMALL, replay_context=False)
    batch = replay_batch()
    other_state = {**batch, "deter": -batch["deter"]}

    metrics = train_metrics(batch, config)

    assert train_metrics(other_state, config)["loss/rec"] == metrics["loss/rec"]


def test_bfloat16_compute_keeps_float32_weights_and_losses():
    agent = make_agent(dataclasses.replace(SMALL, compute_dtype="bfloat16"))

    metrics, states = agent.train(replay_batch(), jax.random.key(1))

    weights = jax.tree.leaves(nnx.state(agent, nnx.Param))
    assert {weight.dtype for weight in weights} == {jnp.dtype(jnp.float32)}
    assert metrics["loss/rec"].dtype == jnp.float32
    assert states.deter.dtype == jnp.bfloat16


N = 3  # The number of environments.


def env_step(is_first):
    """One step of N environments, in the format of `BatchEnv.step`."""
    rng = np.random.default_rng(0)
    return {
        "position": rng.normal(size=(N, 3)).astype(np.float32),
        "velocity": rng.normal(size=(N, 2)).astype(np.float32),
        "reward": np.zeros(N, np.float32),
        "is_first": np.full(N, is_first),
        "is_last": np.zeros(N, bool),
        "is_terminal": np.zeros(N, bool),
    }


def test_policy_gives_one_action_for_each_environment():
    agent = make_agent()

    action, _ = agent.policy(
        agent.initial_policy_state(N), env_step(True), jax.random.key(1)
    )

    assert action.shape == (N, A)
    assert action.dtype == jnp.float32
    assert jnp.isfinite(action).all()


def test_policy_samples_different_actions_with_different_keys():
    agent = make_agent()
    state = agent.initial_policy_state(N)

    action_1, _ = agent.policy(state, env_step(True), jax.random.key(1))
    action_2, _ = agent.policy(state, env_step(True), jax.random.key(2))
    action_3, _ = agent.policy(state, env_step(True), jax.random.key(1))

    assert not jnp.allclose(action_1, action_2)
    np.testing.assert_array_equal(action_1, action_3)


def test_policy_forgets_the_last_episode_at_is_first():
    agent = make_agent()
    state = agent.initial_policy_state(N)
    for key in jax.random.split(jax.random.key(1), 5):
        _, state = agent.policy(state, env_step(False), key)

    after_episode, _ = agent.policy(state, env_step(True), jax.random.key(2))
    fresh, _ = agent.policy(
        agent.initial_policy_state(N), env_step(True), jax.random.key(2)
    )

    np.testing.assert_allclose(after_episode, fresh, atol=1e-6)
