"""The symlog transform.

Rewards, values, and observations can have any scale. For example, one task
gives rewards near 1 and a different task gives rewards near 1000. Large
targets cause large gradients, and these gradients can make training unstable.

DreamerV3 compresses such values with the symlog function:

    symlog(x) = sign(x) * log(|x| + 1)

For small |x|, the function is almost the identity. For large |x|, it grows
like a logarithm. It keeps the sign, so it also works for negative values. The
symexp function is its inverse.

Reference: section "Robust predictions" of the DreamerV3 paper.
"""

import jax
import jax.numpy as jnp


def symlog(x: jax.Array) -> jax.Array:
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x: jax.Array) -> jax.Array:
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))
