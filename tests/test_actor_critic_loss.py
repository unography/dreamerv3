"""The actor and critic losses, and how they learn from imagination."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from dreamer.actor_critic import (
    Actor,
    ReturnNorm,
    actor_loss,
    critic_loss,
    imagination_loss,
    replay_value_loss,
)
from dreamer.distributions import Normal, TwoHot, symexp_bins, two_hot
from dreamer.world_model import TwoHotHead, WorldModel

ONE_STEP = (1, 1)  # One trajectory [N] with one step [H].


def actor_gradient(advantage, entropy_scale=0.0):
    """The gradient of the actor loss with respect to the policy mean and std,
    for the action 1.0 from a policy with mean 0.0 and std 1.0."""

    def loss(mean, std):
        policy = Normal(jnp.full((*ONE_STEP, 1), mean), jnp.full((*ONE_STEP, 1), std))
        actions = jnp.ones((*ONE_STEP, 1))
        advantages = jnp.full(ONE_STEP, advantage)
        weight = jnp.ones(ONE_STEP)
        return actor_loss(policy, actions, advantages, weight, entropy_scale).sum()

    return jax.grad(loss, argnums=(0, 1))(0.0, 1.0)


def test_actor_loss_moves_the_policy_toward_actions_with_a_positive_advantage():
    mean_grad, _ = actor_gradient(advantage=1.0)

    assert mean_grad < 0  # A gradient step increases the mean toward 1.0.


def test_actor_loss_moves_the_policy_away_from_actions_with_a_negative_advantage():
    mean_grad, _ = actor_gradient(advantage=-1.0)

    assert mean_grad > 0


def test_entropy_bonus_widens_the_policy_when_actions_are_equally_good():
    _, std_grad = actor_gradient(advantage=0.0, entropy_scale=0.1)

    assert std_grad < 0  # A gradient step increases the std.


def test_actor_loss_finds_the_best_action_without_a_gradient_of_the_reward():
    # A one-step task: the reward is -(action - 0.7)^2. The actor loss only
    # sees the rewards of its samples, as REINFORCE does.
    def loss(mean, key):
        policy = Normal(jnp.full((256, 1, 1), mean), jnp.full((256, 1, 1), 0.3))
        actions = policy.sample(key)
        rewards = -jnp.square(actions[..., 0] - 0.7)
        advantages = rewards - rewards.mean()
        weight = jnp.ones_like(advantages)
        return actor_loss(policy, actions, advantages, weight, 0.0).mean()

    grad = jax.jit(jax.grad(loss))
    mean = 0.0
    for key in jax.random.split(jax.random.key(0), 300):
        mean -= 0.5 * grad(mean, key)

    np.testing.assert_allclose(mean, 0.7, atol=0.05)


def critic_loss_of(probs, slow_reg):
    """The critic loss of one prediction, for a return of 4 and a slow critic
    value of 0."""
    logits = jnp.log(probs + 1e-6)
    loss = critic_loss(
        TwoHot(logits[None, None], symexp_bins()),
        returns=jnp.full(ONE_STEP, 4.0),
        slow_values=jnp.zeros(ONE_STEP),
        weight=jnp.ones(ONE_STEP),
        slow_reg=slow_reg,
    )
    return float(loss[0, 0])


ON_RETURN = two_hot(jnp.array(4.0), symexp_bins())
ON_SLOW_VALUE = two_hot(jnp.array(0.0), symexp_bins())
HALF_AND_HALF = (ON_RETURN + ON_SLOW_VALUE) / 2


def test_critic_loss_prefers_a_prediction_of_the_return():
    on_return = critic_loss_of(ON_RETURN, slow_reg=0.0)

    assert on_return < critic_loss_of(HALF_AND_HALF, slow_reg=0.0)
    assert on_return < critic_loss_of(ON_SLOW_VALUE, slow_reg=0.0)


def test_slow_critic_term_moves_the_best_prediction_toward_the_slow_critic():
    half_and_half = critic_loss_of(HALF_AND_HALF, slow_reg=1.0)

    assert half_and_half < critic_loss_of(ON_RETURN, slow_reg=1.0)
    assert half_and_half < critic_loss_of(ON_SLOW_VALUE, slow_reg=1.0)


N, A, H = 6, 2, 5  # Start states, action size, imagination horizon.


def make_nets():
    rngs = nnx.Rngs(0)
    world_model = WorldModel(
        {"obs": (3,)},
        A,
        deter=16,
        hidden=8,
        stoch=4,
        classes=3,
        blocks=2,
        units=8,
        rngs=rngs,
    )
    actor = Actor(28, A, units=8, layers=2, rngs=rngs)
    critic = TwoHotHead(28, units=8, layers=2, rngs=rngs)
    slow_critic = TwoHotHead(28, units=8, layers=2, rngs=rngs)
    return world_model, actor, critic, slow_critic


@nnx.jit
def run_imagination_loss(world_model, actor, critic, slow_critic):
    start = world_model.rssm.initial(N)
    return imagination_loss(
        world_model,
        actor,
        critic,
        slow_critic,
        ReturnNorm(),
        start,
        jax.random.key(1),
        horizon=H,
    )


def test_imagination_loss_gives_one_loss_value_for_each_imagined_step():
    losses, returns, _ = run_imagination_loss(*make_nets())

    assert losses["policy"].shape == (N, H)
    assert losses["value"].shape == (N, H)
    assert returns.shape == (N, H)


def test_imagination_loss_trains_the_actor_and_critic_but_not_the_world_model():
    world_model, actor, critic, slow_critic = make_nets()

    def total_loss(world_model, actor, critic, slow_critic):
        losses, _, _ = run_imagination_loss(world_model, actor, critic, slow_critic)
        return sum(loss.mean() for loss in losses.values())

    grad_fn = nnx.jit(nnx.grad(total_loss, argnums=(0, 1, 2)))
    grads = grad_fn(world_model, actor, critic, slow_critic)

    def norm(tree):
        return float(sum(jnp.abs(x).sum() for x in jax.tree.leaves(tree)))

    world_model_grads, actor_grads, critic_grads = grads
    assert norm(world_model_grads) == 0
    assert norm(actor_grads) > 0
    assert norm(critic_grads) > 0


def replay_returns(reward, is_last, is_terminal, bootstrap, lam=0.95):
    """The replay value loss and returns of one trajectory [T]."""
    critic = TwoHotHead(4, units=8, layers=1, rngs=nnx.Rngs(0))
    data = [jnp.asarray(x)[None] for x in (reward, is_last, is_terminal, bootstrap)]
    features = jnp.zeros((1, len(reward), 4))
    loss, returns = replay_value_loss(
        critic, critic, features, *data, horizon=100, lam=lam
    )
    return loss[0], returns[0]


def test_replay_returns_sum_the_real_rewards_and_bootstrap_at_the_end():
    _, returns = replay_returns(
        reward=[0.0, 1.0, 2.0, 3.0],
        is_last=[False] * 4,
        is_terminal=[False] * 4,
        bootstrap=[0.0, 0.0, 0.0, 50.0],
        lam=1.0,
    )

    np.testing.assert_allclose(
        returns[0], 1 + 0.99 * 2 + 0.99**2 * 3 + 0.99**3 * 50, rtol=1e-6
    )


def test_replay_returns_do_not_go_into_the_next_episode():
    # Step 2 is the last step of an episode, so the return of step 1 uses the
    # imagined return of step 2 and not the rewards after it.
    _, returns = replay_returns(
        reward=[0.0, 1.0, 2.0, 100.0],
        is_last=[False, False, True, False],
        is_terminal=[False] * 4,
        bootstrap=[0.0, 0.0, 10.0, 0.0],
    )

    np.testing.assert_allclose(returns[1], 2.0 + 0.99 * 10.0, rtol=1e-6)


def test_replay_value_loss_gives_no_weight_to_the_last_step_of_an_episode():
    loss, _ = replay_returns(
        reward=[0.0, 1.0, 2.0, 3.0],
        is_last=[False, True, False, False],
        is_terminal=[False] * 4,
        bootstrap=[1.0] * 4,
    )

    assert loss[1] == 0
    assert loss[0] > 0
    assert loss[2] > 0
