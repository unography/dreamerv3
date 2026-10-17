"""The properties of symlog that make it a good transform for targets."""

import jax.numpy as jnp
import numpy as np

from dreamer.symlog import symexp, symlog


def test_symexp_is_the_inverse_of_symlog():
    x = jnp.concatenate([-jnp.logspace(-6, 6, 50), jnp.logspace(-6, 6, 50)])

    np.testing.assert_allclose(symexp(symlog(x)), x, rtol=1e-5)


def test_symlog_is_almost_the_identity_for_small_values():
    x = jnp.linspace(-0.01, 0.01, 21)

    np.testing.assert_allclose(symlog(x), x, atol=1e-4)


def test_symlog_compresses_large_values():
    x = jnp.array([10.0, 1e3, 1e6])

    np.testing.assert_allclose(symlog(x), [2.4, 6.9, 13.8], atol=0.05)


def test_symlog_keeps_the_sign():
    x = jnp.logspace(-3, 3, 20)

    np.testing.assert_array_equal(symlog(-x), -symlog(x))
