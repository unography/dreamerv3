"""Actor-critic learning.

The agent learns its behavior from imagined trajectories of the world model.
The critic predicts the return of each state. The actor chooses actions, and
learns to choose actions with high returns.

Reference: sections "Critic learning" and "Actor learning" of the DreamerV3
paper.
"""

import jax
import jax.numpy as jnp


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
