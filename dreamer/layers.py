"""Network layers.

DreamerV3 builds all of its networks from a small set of layers:

- Linear layers. Their weights come from a truncated normal distribution with
  variance 1 / fan_in. This is the default initialization of `nnx.Linear`.
- RMSNorm and the SiLU activation after each hidden linear layer.
- Block-diagonal linear layers in the recurrent model.

Some output layers start with small or zero weights. A head with zero weights
predicts exactly zero at the start of training. `scaled_init` makes these
initializations.

Reference: appendix "Networks" of the DreamerV3 paper.
"""

import itertools

import jax
import jax.numpy as jnp
from flax import nnx


def scaled_init(scale: float) -> nnx.Initializer:
    """The default initialization, with the standard deviation times `scale`."""
    return nnx.initializers.variance_scaling(scale**2, "fan_in", "truncated_normal")


class Dense(nnx.Module):
    """A linear layer, then RMSNorm, then SiLU.

    The norm makes the output scale independent of the input scale. Thus the
    activations stay in a good range for the next layer.
    """

    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(in_features, out_features, rngs=rngs)
        self.norm = nnx.RMSNorm(out_features, epsilon=1e-4, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.nn.silu(self.norm(self.linear(x)))


class MLP(nnx.Module):
    """A stack of `Dense` layers with the same width."""

    def __init__(self, in_features: int, units: int, layers: int, *, rngs: nnx.Rngs):
        sizes = [in_features] + [units] * layers
        pairs = itertools.pairwise(sizes)
        self.layers = nnx.List([Dense(i, o, rngs=rngs) for i, o in pairs])

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers:
            x = layer(x)
        return x


class BlockLinear(nnx.Module):
    """A linear layer with a block-diagonal weight matrix.

    The layer splits its input and its output into `blocks` groups. Each output
    group depends only on the input group with the same index. Thus the layer
    has `blocks` times fewer weights and operations than a dense layer of the
    same size. The recurrent model of DreamerV3 uses this layer, so it can keep
    a large state at a low cost.

    The initialization uses the fan-in of the full input, not of one block, as
    the reference implementation does.
    """

    def __init__(
        self, in_features: int, out_features: int, blocks: int, *, rngs: nnx.Rngs
    ):
        assert in_features % blocks == 0
        assert out_features % blocks == 0
        self.blocks = blocks
        shape = (blocks, in_features // blocks, out_features // blocks)
        self.kernel = nnx.Param(nnx.initializers.lecun_normal()(rngs.params(), shape))
        self.bias = nnx.Param(jnp.zeros(out_features))

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x.reshape(*x.shape[:-1], self.blocks, -1)  # [..., G, I]
        x = jnp.einsum("...gi,gio->...go", x, self.kernel[...])  # [..., G, O]
        return x.reshape(*x.shape[:-2], -1) + self.bias[...]
