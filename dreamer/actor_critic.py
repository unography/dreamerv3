"""Actor-critic learning.

The agent learns its behavior from imagined trajectories of the world model.
The critic predicts the return of each state. The actor chooses actions, and
learns to choose actions with high returns. Both use the model features [h, z]
as their input.

Reference: sections "Critic learning" and "Actor learning" of the DreamerV3
paper.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from dreamer.distributions import Normal, TwoHot, bounded_normal
from dreamer.layers import MLP, scaled_init
from dreamer.rssm import State
from dreamer.world_model import TwoHotHead, WorldModel

sg = jax.lax.stop_gradient


def lambda_return(
    reward: jax.Array,
    value: jax.Array,
    discount: jax.Array,
    lam: float | jax.Array,
) -> jax.Array:
    """Computes the λ-returns of trajectories.

    The λ-return is the target of the critic. It mixes the rewards of the
    trajectory with the values of the critic. The recursion goes backward in
    time and starts from the last value:

        R_t = r_t+1 + γ_t+1 * ((1 - λ_t+1) * v_t+1 + λ_t+1 * R_t+1)
        R_T-1 = v_T-1

    With λ = 0, the return is the one-step target r_t+1 + γ_t+1 * v_t+1. With
    λ = 1, it is the discounted sum of all rewards, plus the discounted last
    value.

    Args:
        reward: [B, T]. Entry t is the reward of the transition into step t.
        value: [B, T]. Entry t is the critic value of step t.
        discount: [B, T]. Entry t is the discount of the transition into step
            t. It is zero for a terminal step, so the return stops there.
        lam: a scalar, or [B, T] with one λ for each step.

    Returns:
        The returns R_0 to R_T-2, as [B, T - 1].
    """
    lam = jnp.broadcast_to(lam, reward.shape)

    def step(next_return, inputs):
        # All inputs are the values of step t+1, as in the equation above.
        r, v, d, l = inputs  # noqa: E741
        ret = r + d * ((1 - l) * v + l * next_return)
        return ret, ret

    # `jax.lax.scan` loops over axis 0, so the inputs are [T - 1, B].
    inputs = [x[:, 1:].T for x in (reward, value, discount, lam)]
    _, returns = jax.lax.scan(step, value[:, -1], inputs, reverse=True)
    return returns.T


class RunningStat(nnx.Variable):
    """A statistic that the training updates directly, not the optimizer."""


class ReturnNorm(nnx.Module):
    """Gives the scale of the returns, for the normalization of advantages.

    The scale is S = max(limit, P_high - P_low), where P_low and P_high are
    percentiles of the returns. Moving averages smooth the two percentiles
    across updates.

    - For large returns, S is the range of the returns. Thus the size of the
      actor gradient does not depend on the reward scale of the task.
    - For small returns, for example with sparse rewards, S is the limit 1.
      Thus the normalization does not make the noise of small returns large.

    Percentiles ignore rare outliers, so one large return does not change S
    much.
    """

    def __init__(
        self,
        rate: float = 0.01,
        limit: float = 1.0,
        low: float = 5.0,
        high: float = 95.0,
    ):
        self.rate = rate
        self.limit = limit
        self.percentiles = (low, high)
        self.low = RunningStat(jnp.zeros(()))
        self.high = RunningStat(jnp.zeros(()))

    def update(self, returns: jax.Array) -> None:
        returns = jax.lax.stop_gradient(returns)
        low, high = jnp.percentile(returns, jnp.array(self.percentiles))
        self.low[...] = (1 - self.rate) * self.low[...] + self.rate * low
        self.high[...] = (1 - self.rate) * self.high[...] + self.rate * high

    def scale(self) -> jax.Array:
        return jnp.maximum(self.limit, self.high[...] - self.low[...])


class Actor(nnx.Module):
    """The policy: a normal distribution over actions for each model state.

    The output layers start with small weights. Thus at the start of training,
    the policy has a mean near 0 and a large standard deviation in each state.
    """

    def __init__(
        self,
        feature_size: int,
        action_size: int,
        *,
        units: int = 64,
        layers: int = 3,
        min_std: float = 0.1,
        max_std: float = 1.0,
        rngs: nnx.Rngs,
    ):
        self.min_std = min_std
        self.max_std = max_std
        self.mlp = MLP(feature_size, units, layers, rngs=rngs)
        init = scaled_init(0.01)
        self.mean = nnx.Linear(units, action_size, kernel_init=init, rngs=rngs)
        self.std = nnx.Linear(units, action_size, kernel_init=init, rngs=rngs)

    def __call__(self, features: jax.Array) -> Normal:
        x = self.mlp(features)
        return bounded_normal(self.mean(x), self.std(x), self.min_std, self.max_std)


def update_slow_critic(
    slow_critic: nnx.Module, critic: nnx.Module, rate: float = 0.02
) -> None:
    """Moves the weights of the slow critic toward the weights of the critic.

    The slow critic is an exponential moving average of the critic. Its
    predictions change slowly, so the critic loss can use them as a stable
    target.
    """
    slow = nnx.state(slow_critic, nnx.Param)
    fast = nnx.state(critic, nnx.Param)
    nnx.update(slow_critic, jax.tree.map(lambda s, f: s + rate * (f - s), slow, fast))


def actor_loss(
    policy: Normal,
    actions: jax.Array,
    advantages: jax.Array,
    weight: jax.Array,
    entropy_scale: float = 3e-4,
) -> jax.Array:
    """The loss of the actor: REINFORCE with an entropy bonus.

    The loss increases the log probability of actions with a positive
    advantage, and decreases the log probability of actions with a negative
    advantage. The gradient does not go through the world model. The
    entropy bonus keeps the policy random, so the agent continues to explore.
    The advantages are normalized, so one entropy scale works for all tasks.

    Args:
        policy: the distributions [N, H, A] of the actor.
        actions: the imagined actions [N, H, A].
        advantages: the normalized advantages [N, H].
        weight: the weight of each step [N, H].
        entropy_scale: the scale of the entropy bonus.

    Returns:
        The loss [N, H].
    """
    log_prob = policy.log_prob(sg(actions)).sum(-1)
    entropy = policy.entropy().sum(-1)
    return -sg(weight) * (log_prob * sg(advantages) + entropy_scale * entropy)


def critic_loss(
    critic: TwoHot,
    returns: jax.Array,
    slow_values: jax.Array,
    weight: jax.Array,
    slow_reg: float = 1.0,
) -> jax.Array:
    """The loss of the critic.

    The critic learns the λ-returns with the two-hot loss. A second loss term
    pulls the critic toward the predictions of the slow critic. The slow
    critic changes slowly, so this term makes the critic training stable.

    Args:
        critic: the distributions [N, H] of the critic.
        returns: the λ-returns [N, H].
        slow_values: the predictions [N, H] of the slow critic.
        weight: the weight of each step [N, H].
        slow_reg: the scale of the slow critic term.

    Returns:
        The loss [N, H].
    """
    loss = critic.loss(sg(returns)) + slow_reg * critic.loss(sg(slow_values))
    return sg(weight) * loss


def imagination_loss(
    world_model: WorldModel,
    actor: Actor,
    critic: TwoHotHead,
    slow_critic: TwoHotHead,
    return_norm: ReturnNorm,
    start: State,
    key: jax.Array,
    *,
    horizon: int = 15,
    lam: float = 0.95,
    entropy_scale: float = 3e-4,
    slow_reg: float = 1.0,
) -> tuple[dict[str, jax.Array], jax.Array, dict[str, jax.Array]]:
    """The losses of the actor and the critic on imagined trajectories.

    The world model imagines trajectories of `horizon` steps from each start
    state. The heads of the world model predict the rewards and the
    discounts along the trajectories. The λ-returns of these predictions are
    the targets of the critic, and give the advantages of the actor.

    Each step has a weight: the product of the predicted discounts up to that
    step. Thus steps after a predicted end of the episode count less.

    The world model does not learn from these losses. It only learns to
    predict the environment.

    Args:
        world_model: gives the imagined trajectories and their rewards.
        actor: the policy that chooses the imagined actions.
        critic: the critic that the loss trains.
        slow_critic: the moving average of the critic.
        return_norm: the scale of the returns. The loss updates it.
        start: the start states [N, ...], usually the posterior states of
            replay data.
        key: the random key for the imagination.

    Returns:
        The losses `policy` and `value`, each [N, H], the λ-returns [N, H],
        and metrics.
    """

    def act(state, key):
        return actor(state.features()).sample(key)

    states, actions = world_model.rssm.imagine(start, act, horizon, key)
    states = sg(states)  # The world model does not learn from these losses.
    features = states.features()  # [N, H + 1, F]
    reward = sg(world_model.reward(features).mean())  # [N, H + 1]
    discount = sg(world_model.cont(features).prob())  # [N, H + 1]
    value = sg(critic(features).mean())  # [N, H + 1]

    returns = lambda_return(reward, value, discount, lam)  # [N, H]
    weight = jnp.cumprod(discount, axis=1)[:, :-1]  # [N, H]
    return_norm.update(returns)
    advantages = (returns - value[:, :-1]) / return_norm.scale()

    # The last state has no return, so the losses use the first H states.
    features = features[:, :-1]
    policy = actor(features)
    losses = {
        "policy": actor_loss(policy, actions, advantages, weight, entropy_scale),
        "value": critic_loss(
            critic(features),
            returns,
            slow_critic(features).mean(),
            weight,
            slow_reg,
        ),
    }
    metrics = {
        "return": returns.mean(),
        "return_scale": return_norm.scale(),
        "advantage": advantages.mean(),
        "advantage_std": advantages.std(),
        "policy_entropy": policy.entropy().sum(-1).mean(),
        "policy_std": policy.std.mean(),
        "imagined_reward": reward.mean(),
        "imagined_discount": discount.mean(),
    }
    return losses, returns, metrics
