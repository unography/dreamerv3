"""Output distributions of the network heads.

Each head gives the parameters of a distribution. The distribution then gives
predictions, samples, and losses.
"""

import jax
import jax.numpy as jnp

from dreamer.symlog import symexp


def symexp_bins(num: int = 255, limit: float = 20.0) -> jax.Array:
    """Gives `num` bins that have equal spaces in symlog space.

    The bins go from -symexp(limit) to symexp(limit). The center bin is exactly
    zero, and each negative bin is the exact negative of a positive bin.
    """
    assert num % 2 == 1, "The bins must have a center bin at zero."
    negative = symexp(jnp.linspace(-limit, 0.0, (num + 1) // 2))
    return jnp.concatenate([negative, -negative[:-1][::-1]])


def two_hot(x: jax.Array, bins: jax.Array) -> jax.Array:
    """Encodes values [...] as weights [..., K] on the two nearest bins.

    The weights interpolate linearly between the two bins, so the weighted sum
    of the bins is equal to x. A value outside the bins puts all of its weight
    on the nearest edge bin.
    """
    x = jnp.clip(x, bins[0], bins[-1])
    num_below = jnp.sum(bins <= x[..., None], axis=-1)
    above = jnp.clip(num_below, 1, len(bins) - 1)
    below = above - 1
    weight_above = (x - bins[below]) / (bins[above] - bins[below])
    return (
        jax.nn.one_hot(below, len(bins)) * (1 - weight_above)[..., None]
        + jax.nn.one_hot(above, len(bins)) * weight_above[..., None]
    )


class TwoHot:
    """A categorical distribution over fixed bins that predicts a scalar.

    DreamerV3 uses it for rewards and values. The network gives logits over
    the bins. The prediction is the mean of the bins under the softmax
    probabilities. The loss is the cross-entropy between the probabilities
    and the two-hot encoding of the target.

    The gradient of this loss with respect to the logits is the difference of
    two probability vectors. Thus the gradient stays small, also for very
    large targets. With a squared error, the gradient increases with the
    error.

    Reference: section "Robust predictions" of the DreamerV3 paper.
    """

    def __init__(self, logits: jax.Array, bins: jax.Array):
        assert logits.shape[-1] == len(bins)
        assert len(bins) % 2 == 1, "The mean needs a center bin at zero."
        self.logits = logits
        self.bins = bins

    def mean(self) -> jax.Array:
        probs = jax.nn.softmax(self.logits)
        center = len(self.bins) // 2
        # Pair each negative bin with its positive bin before the sum. If the
        # two probabilities are equal, the two terms cancel exactly. Thus a
        # head with uniform probabilities predicts exactly zero. A plain sum
        # from left to right adds rounding errors instead.
        negative = (probs[..., :center] * self.bins[:center])[..., ::-1]
        positive = probs[..., center + 1 :] * self.bins[center + 1 :]
        return (negative + positive).sum(-1) + probs[..., center] * self.bins[center]

    def loss(self, target: jax.Array) -> jax.Array:
        target = jax.lax.stop_gradient(two_hot(target, self.bins))
        return -(target * jax.nn.log_softmax(self.logits)).sum(-1)
