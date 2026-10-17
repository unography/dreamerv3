"""The optimizer of DreamerV3.

The optimizer is a chain of three steps:

1. Adaptive gradient clipping (AGC). For each weight array, AGC limits the
   gradient norm to a fraction of the weight norm. Thus one update cannot
   change a layer much, and a rare large gradient does not damage the weights.
2. LaProp. It divides each gradient element by the root mean square of its
   recent values, as RMSProp does. Then it takes the momentum of these
   normalized gradients. Adam does the same two operations in the opposite
   order. The LaProp order stays stable with a very small epsilon.
3. A learning rate with a linear warmup over the first updates.

Reference: appendix "Optimizer" of the DreamerV3 paper.
"""

import jax
import jax.numpy as jnp
import optax


def clip_by_agc(
    clip: float = 0.3, min_norm: float = 1e-3
) -> optax.GradientTransformation:
    """Limits the norm of each gradient array to `clip` times its weight norm.

    The weight norm has a minimum of `min_norm`, so weights near zero can
    still change.
    """

    def init(params):
        return optax.EmptyState()

    def update(updates, state, params):
        def clip_array(update, param):
            limit = clip * jnp.maximum(min_norm, jnp.linalg.norm(param))
            return update / jnp.maximum(1.0, jnp.linalg.norm(update) / limit)

        return jax.tree.map(clip_array, updates, params), state

    return optax.GradientTransformation(init, update)


def make_optimizer(
    lr: float = 4e-5,
    agc: float = 0.3,
    eps: float = 1e-20,
    beta1: float = 0.9,
    beta2: float = 0.999,
    warmup: int = 1000,
) -> optax.GradientTransformation:
    """Makes the DreamerV3 optimizer: AGC, then LaProp, then the warmup."""
    schedule = optax.linear_schedule(0.0, lr, warmup) if warmup else lr
    return optax.chain(
        clip_by_agc(agc),
        # LaProp: the normalization by the root mean square, then the momentum.
        optax.scale_by_rms(
            decay=beta2, eps=eps, eps_in_sqrt=False, bias_correction=True
        ),
        optax.ema(decay=beta1, debias=True),
        optax.scale_by_learning_rate(schedule),
    )
