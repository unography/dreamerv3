"""λ-returns: the targets of the critic."""

import jax.numpy as jnp
import numpy as np
import pytest

from dreamer.actor_critic import lambda_return

REWARD = jnp.array([[0.0, 1.0, 2.0, 3.0]])
VALUE = jnp.array([[10.0, 20.0, 30.0, 40.0]])
DISCOUNT = jnp.full((1, 4), 0.9)


def test_lambda_zero_gives_one_step_targets():
    returns = lambda_return(REWARD, VALUE, DISCOUNT, lam=0.0)

    expected = REWARD[:, 1:] + 0.9 * VALUE[:, 1:]
    np.testing.assert_allclose(returns, expected)


def test_lambda_one_gives_the_discounted_rewards_and_the_last_value():
    returns = lambda_return(REWARD, VALUE, DISCOUNT, lam=1.0)

    first = 1.0 + 0.9 * 2.0 + 0.9**2 * 3.0 + 0.9**3 * 40.0
    np.testing.assert_allclose(returns[0, 0], first, rtol=1e-6)


def test_zero_discount_at_a_terminal_step_stops_the_return():
    discount = DISCOUNT.at[:, 2].set(0.0)  # Step 2 is terminal.

    returns = lambda_return(REWARD, VALUE, discount, lam=1.0)

    # The return of step 1 is only the reward of the move into step 2.
    np.testing.assert_allclose(returns[0, 1], 2.0)
    np.testing.assert_allclose(returns[0, 0], 1.0 + 0.9 * 2.0)


@pytest.mark.parametrize("lam", [0.0, 0.5, 0.95, 1.0])
def test_exact_values_give_the_same_return_for_any_lambda(lam):
    # With a reward of 1 at each step forever, the true value of each step is
    # 1 / (1 - 0.9) = 10. All λ-returns then agree with the values.
    reward = jnp.ones((1, 6))
    value = jnp.full((1, 6), 10.0)
    discount = jnp.full((1, 6), 0.9)

    returns = lambda_return(reward, value, discount, lam)

    np.testing.assert_allclose(returns, 10.0, rtol=1e-6)


def test_lambda_can_change_at_each_step():
    # A step with λ = 0 starts a new return. The replay value loss uses this at
    # the last step of an episode.
    lam = jnp.array([[1.0, 1.0, 0.0, 1.0]])

    returns = lambda_return(REWARD, VALUE, DISCOUNT, lam)

    np.testing.assert_allclose(returns[0, 1], 2.0 + 0.9 * 30.0)
