"""The world model.

The world model learns from replay data how the environment works. It has
these parts:

- The encoder maps each observation to a token e_t.
- The RSSM (see `rssm.py`) keeps a model state from the tokens and actions.
- The decoder reconstructs the observation from the model state. This loss
  makes the state hold the information of the observations.

The model state is the input of all heads. Thus the agent can imagine the
future with the RSSM prior, and get the predictions of the heads for these
imagined states.

Reference: section "World model learning" of the DreamerV3 paper.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from dreamer.layers import MLP
from dreamer.symlog import symlog


class Encoder(nnx.Module):
    """Maps the observation vectors of one step to a token e_t.

    The encoder applies symlog to the inputs. Thus an observation with large
    values does not give large activations in the first layer.
    """

    def __init__(
        self,
        obs_shapes: dict[str, tuple[int]],
        *,
        units: int = 64,
        layers: int = 3,
        rngs: nnx.Rngs,
    ):
        self.keys = sorted(obs_shapes)
        size = sum(obs_shapes[key][0] for key in self.keys)
        self.mlp = MLP(size, units, layers, rngs=rngs)

    def __call__(self, obs: dict[str, jax.Array]) -> jax.Array:
        x = jnp.concatenate([obs[key] for key in self.keys], -1)
        return self.mlp(symlog(x))


class Decoder(nnx.Module):
    """Predicts the observation vectors from the model features [h, z].

    The predictions are in symlog space. See `symlog_mse`.
    """

    def __init__(
        self,
        feature_size: int,
        obs_shapes: dict[str, tuple[int]],
        *,
        units: int = 64,
        layers: int = 3,
        rngs: nnx.Rngs,
    ):
        self.keys = sorted(obs_shapes)
        self.sizes = [obs_shapes[key][0] for key in self.keys]
        self.mlp = MLP(feature_size, units, layers, rngs=rngs)
        self.out = nnx.Linear(units, sum(self.sizes), rngs=rngs)

    def __call__(self, features: jax.Array) -> dict[str, jax.Array]:
        x = self.out(self.mlp(features))
        splits = np.cumsum(self.sizes)[:-1]
        return dict(zip(self.keys, jnp.split(x, splits, -1), strict=True))


def symlog_mse(prediction: jax.Array, target: jax.Array) -> jax.Array:
    """The squared error in symlog space, summed over the last axis.

    For targets near zero, the loss is almost the usual squared error. For
    large targets, the loss is much smaller. Thus observations with large
    values do not dominate the loss.
    """
    return jnp.square(prediction - symlog(target)).sum(-1)
