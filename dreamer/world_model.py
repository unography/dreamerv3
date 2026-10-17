"""The world model.

The world model learns from replay data how the environment works. It has
these parts:

- The encoder maps each observation to a token e_t.
- The RSSM (see `rssm.py`) keeps a model state from the tokens and actions.
- The decoder reconstructs the observation from the model state. This loss
  makes the state hold the information of the observations.

- The reward head predicts the reward of each step.
- The continue head predicts if the episode continues after each step.

The model state is the input of all heads. Thus the agent can imagine the
future with the RSSM prior, and get the predictions of the heads for these
imagined states.

Reference: section "World model learning" of the DreamerV3 paper.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from dreamer.distributions import Bernoulli, TwoHot, symexp_bins
from dreamer.layers import MLP, scaled_init
from dreamer.rssm import RSSM, State
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


class TwoHotHead(nnx.Module):
    """An MLP with a two-hot output. The reward head and the critic use it.

    The output layer starts with zero weights, so the head predicts exactly
    zero at the start of training. Thus the first predictions of rewards and
    values are not large random numbers.
    """

    def __init__(
        self,
        in_features: int,
        *,
        units: int = 64,
        layers: int = 1,
        bins: int = 255,
        rngs: nnx.Rngs,
    ):
        self.bins = bins
        self.mlp = MLP(in_features, units, layers, rngs=rngs)
        self.out = nnx.Linear(units, bins, kernel_init=scaled_init(0.0), rngs=rngs)

    def __call__(self, features: jax.Array) -> TwoHot:
        return TwoHot(self.out(self.mlp(features)), symexp_bins(self.bins))


class ContinueHead(nnx.Module):
    """An MLP that predicts if the episode continues after a step."""

    def __init__(
        self, in_features: int, *, units: int = 64, layers: int = 1, rngs: nnx.Rngs
    ):
        self.mlp = MLP(in_features, units, layers, rngs=rngs)
        self.out = nnx.Linear(units, 1, rngs=rngs)

    def __call__(self, features: jax.Array) -> Bernoulli:
        return Bernoulli(self.out(self.mlp(features))[..., 0])


def continue_target(is_terminal: jax.Array, horizon: float = 333) -> jax.Array:
    """The target of the continue head: the discount factor of each step.

    The target is 1 - 1 / horizon for steps that continue, and zero for
    terminal steps. Thus the continue head learns the discount factor and the
    episode ends together. The imagined returns then need no other discount.
    """
    return (1 - is_terminal.astype(jnp.float32)) * (1 - 1 / horizon)


class WorldModel(nnx.Module):
    """The encoder, the RSSM, the decoder, and the reward and continue heads.

    The size arguments have the defaults of the smallest DreamerV3 model, which
    the reference uses for DMC tasks with proprioceptive observations.
    """

    def __init__(
        self,
        obs_shapes: dict[str, tuple[int]],
        action_size: int,
        *,
        deter: int = 512,
        hidden: int = 64,
        stoch: int = 32,
        classes: int = 4,
        blocks: int = 8,
        units: int = 64,
        unimix: float = 0.01,
        rngs: nnx.Rngs,
    ):
        self.obs_keys = sorted(obs_shapes)
        feature_size = deter + stoch * classes
        self.encoder = Encoder(obs_shapes, units=units, layers=3, rngs=rngs)
        self.rssm = RSSM(
            units,
            action_size,
            deter=deter,
            hidden=hidden,
            stoch=stoch,
            classes=classes,
            blocks=blocks,
            unimix=unimix,
            rngs=rngs,
        )
        self.decoder = Decoder(
            feature_size, obs_shapes, units=units, layers=3, rngs=rngs
        )
        self.reward = TwoHotHead(feature_size, units=units, layers=1, rngs=rngs)
        self.cont = ContinueHead(feature_size, units=units, layers=1, rngs=rngs)

    def loss(
        self,
        start: State,
        prev_actions: jax.Array,
        data: dict[str, jax.Array],
        key: jax.Array,
        *,
        horizon: float = 333,
        free_nats: float = 1.0,
    ) -> tuple[dict[str, jax.Array], State]:
        """Gives the world model losses on a batch of replay sequences.

        Args:
            start: the state before the sequences. Shapes [B, ...].
            prev_actions: the action before each step [B, T, A].
            data: the observations, rewards, and flags of the steps [B, T, ...].
            key: the random key for the samples of z.

        Returns:
            The losses, each [B, T], and the posterior states [B, T, ...]:

            - `dyn`, `rep`: the KL losses of the RSSM.
            - `rec`: the reconstruction loss of the observations.
            - `rew`: the reward loss.
            - `con`: the continue loss.
        """
        tokens = self.encoder({name: data[name] for name in self.obs_keys})
        states, post_logits = self.rssm.observe(
            start, prev_actions, tokens, data["is_first"], key
        )
        dyn, rep = self.rssm.kl_losses(states.deter, post_logits, free_nats)
        features = states.features()
        recons = self.decoder(features)
        losses = {
            "dyn": dyn,
            "rep": rep,
            "rec": sum(symlog_mse(recons[k], data[k]) for k in self.obs_keys),
            "rew": self.reward(features).loss(data["reward"]),
            "con": -self.cont(features).log_prob(
                continue_target(data["is_terminal"], horizon)
            ),
        }
        return losses, states
