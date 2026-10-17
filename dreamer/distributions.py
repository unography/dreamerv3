"""Output distributions of the network heads.

Each head gives the parameters of a distribution. The distribution then gives
predictions, samples, and losses. The distributions compute in float32, also
when the networks compute in bfloat16, so the losses stay accurate.
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
        self.logits = logits.astype(jnp.float32)
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


class OneHotCategorical:
    """A categorical distribution with one-hot samples.

    The stochastic state of the world model is a vector of these
    distributions. The distribution uses two methods from DreamerV3:

    - Unimix: the probabilities are a mix of 99% softmax and 1% uniform. Thus
      no class has a probability near zero, and the KL divergence between two
      distributions has an upper limit. Without this limit, the KL losses can
      have large spikes.
    - Straight-through gradients: a one-hot sample has no gradient. Thus a
      sample uses the gradient of the probabilities instead.

    Reference: section "World model learning" and appendix "Distributions"
    of the DreamerV3 paper.
    """

    def __init__(self, logits: jax.Array, unimix: float = 0.0):
        logits = logits.astype(jnp.float32)
        if unimix:
            probs = jax.nn.softmax(logits)
            probs = (1 - unimix) * probs + unimix / logits.shape[-1]
            logits = jnp.log(probs)
        self.log_probs = jax.nn.log_softmax(logits)
        self.probs = jnp.exp(self.log_probs)

    def sample(self, key: jax.Array) -> jax.Array:
        index = jax.random.categorical(key, self.log_probs)
        onehot = jax.nn.one_hot(index, self.probs.shape[-1], dtype=self.probs.dtype)
        # The value is the one-hot vector. The gradient is the gradient of the
        # probabilities. The parentheses make the value exact: they add zero.
        return onehot + (self.probs - jax.lax.stop_gradient(self.probs))

    def entropy(self) -> jax.Array:
        return -(self.probs * self.log_probs).sum(-1)

    def kl(self, other: "OneHotCategorical") -> jax.Array:
        """The KL divergence KL(self || other)."""
        return (self.probs * (self.log_probs - other.log_probs)).sum(-1)


class Normal:
    """An independent normal distribution for each action dimension."""

    def __init__(self, mean: jax.Array, std: jax.Array):
        self.mean = jnp.asarray(mean, jnp.float32)
        self.std = jnp.asarray(std, jnp.float32)

    def sample(self, key: jax.Array) -> jax.Array:
        return self.mean + self.std * jax.random.normal(key, self.mean.shape)

    def log_prob(self, x: jax.Array) -> jax.Array:
        z = (x - self.mean) / self.std
        return -0.5 * jnp.square(z) - jnp.log(self.std) - 0.5 * jnp.log(2 * jnp.pi)

    def entropy(self) -> jax.Array:
        return 0.5 * jnp.log(2 * jnp.pi * jnp.e * jnp.square(self.std))


def bounded_normal(
    mean: jax.Array, std: jax.Array, min_std: float = 0.1, max_std: float = 1.0
) -> Normal:
    """Makes the DreamerV3 policy distribution for continuous actions.

    The inputs are raw network outputs. A tanh puts the mean into [-1, 1], the
    action range. A sigmoid puts the standard deviation into the range
    [min_std, max_std]. The offset of 2 in the sigmoid moves the initial
    standard deviation near the maximum, so the policy explores much at the
    start of training.

    The samples are not bounded. The environment clips them to [-1, 1].
    """
    mean, std = mean.astype(jnp.float32), std.astype(jnp.float32)
    std = (max_std - min_std) * jax.nn.sigmoid(std + 2.0) + min_std
    return Normal(jnp.tanh(mean), std)


class Bernoulli:
    """A distribution over 0 and 1, with the probability of 1 from a logit."""

    def __init__(self, logit: jax.Array):
        self.logit = logit.astype(jnp.float32)

    def prob(self) -> jax.Array:
        """The probability of 1."""
        return jax.nn.sigmoid(self.logit)

    def log_prob(self, x: jax.Array) -> jax.Array:
        """The log probability of x.

        A value of x between 0 and 1 is a soft target. The result is then the
        negative cross-entropy between x and the predicted probability.
        """
        log_p1 = jax.nn.log_sigmoid(self.logit)
        log_p0 = jax.nn.log_sigmoid(-self.logit)
        return x * log_p1 + (1 - x) * log_p0
