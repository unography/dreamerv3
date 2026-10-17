"""The training step and the policy of the agent."""

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
    batch = {
        "position": rng.normal(size=(B, T, 3)).astype(np.float32),
        "velocity": rng.normal(size=(B, T, 2)).astype(np.float32),
        "action": rng.uniform(-1, 1, size=(B, T, A)).astype(np.float32),
        "reward": rng.uniform(size=(B, T)).astype(np.float32),
        "is_first": np.zeros((B, T), bool),
        "is_last": np.zeros((B, T), bool),
        "is_terminal": np.zeros((B, T), bool),
    }
    batch["is_first"][:, 0] = True
    return batch


def copy_of_weights(module):
    return jax.tree.map(np.array, nnx.state(module, nnx.Param))


def test_train_step_gives_finite_metrics():
    metrics = make_agent().train(replay_batch(), jax.random.key(1))

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
    first = agent.train(batch, keys[0])
    for key in keys[1:]:
        last = agent.train(batch, key)

    assert last["loss/rec"] < 0.8 * first["loss/rec"]
    assert last["loss/con"] < 0.8 * first["loss/con"]
