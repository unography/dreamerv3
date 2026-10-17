"""Recurrent state-space model (RSSM).

The RSSM is the core of the world model. It keeps a model state that has two
parts:

- `deter`: a deterministic vector h. A recurrent network updates it from the
  last state and the last action. It is the memory of the model.
- `stoch`: a stochastic vector z. It is a set of categorical variables, each a
  one-hot vector. It holds the information of the current step.

For each step, the model computes two distributions for z:

    h_t = f(h_t-1, z_t-1, a_t-1)    recurrent step
    z_t ~ q(z_t | h_t, e_t)         posterior: uses the observation token e_t
    z_t ~ p(z_t | h_t)              prior: does not use the observation

The model uses the posterior when it has observations, for example on replay
data. It uses the prior to imagine the future without observations. The
training makes the prior predict the posterior, so imagination becomes
accurate.

Reference: section "World model learning" of the DreamerV3 paper, and
`dreamerv3/rssm.py` of the reference implementation.
"""

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx

from dreamer.distributions import OneHotCategorical
from dreamer.layers import MLP, BlockLinear, Dense


class State(NamedTuple):
    """The state of the RSSM."""

    deter: jax.Array  # [..., D]
    stoch: jax.Array  # [..., S, C], one-hot vectors.

    def features(self) -> jax.Array:
        """The input of the heads: [h, z] as one flat vector [..., D + S * C]."""
        stoch = self.stoch.reshape(*self.stoch.shape[:-2], -1)
        return jnp.concatenate([self.deter, stoch], -1)


class RSSM(nnx.Module):
    """The recurrent state-space model of DreamerV3.

    Args:
        token_size: the size of an encoded observation.
        action_size: the size of an action.
        deter: the size of the deterministic state h.
        hidden: the width of the hidden layers.
        stoch: the number of categorical variables in z.
        classes: the number of classes of each categorical variable.
        blocks: the number of blocks of the recurrent network.
        unimix: the part of the uniform distribution in each categorical.
    """

    def __init__(
        self,
        token_size: int,
        action_size: int,
        *,
        deter: int = 512,
        hidden: int = 64,
        stoch: int = 32,
        classes: int = 4,
        blocks: int = 8,
        unimix: float = 0.01,
        rngs: nnx.Rngs,
    ):
        self.deter_size = deter
        self.stoch_size = stoch
        self.classes = classes
        self.blocks = blocks
        self.unimix = unimix

        # The recurrent network: a GRU with block-diagonal weights.
        self.deter_in = Dense(deter, hidden, rngs=rngs)
        self.stoch_in = Dense(stoch * classes, hidden, rngs=rngs)
        self.action_in = Dense(action_size, hidden, rngs=rngs)
        gru_in = deter + blocks * 3 * hidden
        self.gru_hidden = BlockLinear(gru_in, deter, blocks, rngs=rngs)
        self.gru_norm = nnx.RMSNorm(deter, epsilon=1e-4, rngs=rngs)
        self.gru_gates = BlockLinear(deter, 3 * deter, blocks, rngs=rngs)

        # The posterior q(z | h, e) and the prior p(z | h).
        self.posterior_hidden = Dense(deter + token_size, hidden, rngs=rngs)
        self.posterior_logits = nnx.Linear(hidden, stoch * classes, rngs=rngs)
        self.prior_hidden = MLP(deter, hidden, layers=2, rngs=rngs)
        self.prior_logits = nnx.Linear(hidden, stoch * classes, rngs=rngs)

    def initial(self, batch_size: int) -> State:
        """The state before the first step of an episode: all zeros."""
        return State(
            deter=jnp.zeros((batch_size, self.deter_size)),
            stoch=jnp.zeros((batch_size, self.stoch_size, self.classes)),
        )

    def dist(self, logits: jax.Array) -> OneHotCategorical:
        return OneHotCategorical(logits, self.unimix)

    def obs_step(
        self,
        state: State,
        action: jax.Array,
        token: jax.Array,
        is_first: jax.Array,
        key: jax.Array,
    ) -> tuple[State, jax.Array]:
        """Steps the model with an observation token e_t.

        Args:
            state: the last state (h_t-1, z_t-1). Shapes [B, ...].
            action: the last action a_t-1 [B, A].
            token: the encoded observation e_t [B, E].
            is_first: [B]. True resets the last state and action to zero, so
                the episode starts from the initial state.
            key: the random key for the sample of z_t.

        Returns:
            The new state (h_t, z_t) and the posterior logits [B, S, C].
        """
        keep = ~is_first
        state = jax.tree.map(lambda x: x * _expand(keep, x), state)
        action = action * _expand(keep, action)
        deter = self.recurrent(state, action)
        logits = self.posterior(deter, token)
        stoch = self.dist(logits).sample(key)
        return State(deter, stoch), logits

    def img_step(
        self, state: State, action: jax.Array, key: jax.Array
    ) -> tuple[State, jax.Array]:
        """Steps the model without an observation, with a sample of the prior.

        Returns:
            The new state (h_t, z_t) and the prior logits [B, S, C].
        """
        deter = self.recurrent(state, action)
        logits = self.prior(deter)
        stoch = self.dist(logits).sample(key)
        return State(deter, stoch), logits

    def observe(
        self,
        state: State,
        prev_actions: jax.Array,
        tokens: jax.Array,
        is_first: jax.Array,
        key: jax.Array,
    ) -> tuple[State, jax.Array]:
        """Runs `obs_step` along a sequence of observations.

        Args:
            state: the state before the sequence. Shapes [B, ...].
            prev_actions: [B, T, A]. Entry t is the action a_t-1 before step t.
            tokens: the encoded observations [B, T, E].
            is_first: [B, T].
            key: the random key for the samples of z.

        Returns:
            The states [B, T, ...] and the posterior logits [B, T, S, C].
        """

        def step(state, inputs):
            action, token, first, key = inputs
            state, logits = self.obs_step(state, action, token, first, key)
            return state, (state, logits)

        inputs = _swap_batch_and_time((prev_actions, tokens, is_first))
        keys = jax.random.split(key, tokens.shape[1])
        _, outputs = jax.lax.scan(step, state, (*inputs, keys))
        return _swap_batch_and_time(outputs)

    def imagine(
        self,
        start: State,
        policy: Callable[[State, jax.Array], jax.Array],
        horizon: int,
        key: jax.Array,
    ) -> tuple[State, jax.Array]:
        """Imagines trajectories with the prior and a policy.

        Args:
            start: the start states. Shapes [B, ...].
            policy: a function (state, key) -> action [B, A].
            horizon: the number H of imagined steps.
            key: the random key for the actions and the samples of z.

        Returns:
            The states [B, H + 1, ...], which begin with the start states, and
            the actions [B, H]. Action t moves the model from state t to
            state t + 1.
        """

        def step(state, key):
            policy_key, step_key = jax.random.split(key)
            action = policy(state, policy_key)
            state, _ = self.img_step(state, action, step_key)
            return state, (state, action)

        keys = jax.random.split(key, horizon)
        _, (states, actions) = jax.lax.scan(step, start, keys)
        states = jax.tree.map(
            lambda first, rest: jnp.concatenate([first[None], rest]), start, states
        )
        return _swap_batch_and_time((states, actions))

    def kl_losses(
        self, deter: jax.Array, post_logits: jax.Array, free_nats: float = 1.0
    ) -> tuple[jax.Array, jax.Array]:
        """Gives the two KL losses that train the prior and the posterior.

        Both losses are the KL divergence KL(q || p) between the posterior q
        and the prior p. Each loss stops the gradient on a different side:

        - The dynamics loss KL(sg(q) || p) trains the prior to predict the
          posterior.
        - The representation loss KL(q || sg(p)) trains the posterior to stay
          predictable. The world model gives it a smaller scale, so the
          posterior keeps the information about the observations.

        Free bits: each loss has a minimum of `free_nats`. Below this value,
        the loss is constant and has no gradient. Thus the model does not try
        to make good predictions perfect, and the other losses get the
        capacity of the model.

        Args:
            deter: the deterministic states h [B, T, D] from `observe`.
            post_logits: the posterior logits [B, T, S, C] from `observe`.
            free_nats: the minimum of each loss.

        Returns:
            The dynamics loss and the representation loss, each [B, T]. Each
            loss is a sum over the S categorical variables.
        """
        sg = jax.lax.stop_gradient
        prior_logits = self.prior(deter)
        dyn = self.dist(sg(post_logits)).kl(self.dist(prior_logits)).sum(-1)
        rep = self.dist(post_logits).kl(self.dist(sg(prior_logits))).sum(-1)
        return jnp.maximum(dyn, free_nats), jnp.maximum(rep, free_nats)

    def recurrent(self, state: State, action: jax.Array) -> jax.Array:
        """Computes the next deterministic state h_t = f(h_t-1, z_t-1, a_t-1)."""
        deter = state.deter  # [B, D]
        stoch = state.stoch.reshape(*state.stoch.shape[:-2], -1)  # [B, S * C]
        # The environment clips actions to [-1, 1], but the samples of the
        # policy can be outside of this range. Thus the model also clips them.
        action = action / jax.lax.stop_gradient(jnp.maximum(1, jnp.abs(action)))

        inputs = [self.deter_in(deter), self.stoch_in(stoch), self.action_in(action)]
        inputs = jnp.concatenate(inputs, -1)  # [B, 3 * H]
        # Each block of the GRU gets its own part of h and all of the inputs.
        g = self.blocks
        x = jnp.concatenate(
            [
                deter.reshape(*deter.shape[:-1], g, -1),  # [B, G, D / G]
                jnp.repeat(inputs[..., None, :], g, -2),  # [B, G, 3 * H]
            ],
            -1,
        )
        x = x.reshape(*x.shape[:-2], -1)
        x = jax.nn.silu(self.gru_norm(self.gru_hidden(x)))  # [B, D]

        # Each block makes its own three gates.
        gates = self.gru_gates(x).reshape(*deter.shape[:-1], g, 3, -1)
        reset, candidate, update = [
            gate.reshape(deter.shape) for gate in jnp.unstack(gates, axis=-2)
        ]
        reset = jax.nn.sigmoid(reset)
        candidate = jnp.tanh(reset * candidate)
        # The offset of -1 keeps most of the old state at the start of training.
        update = jax.nn.sigmoid(update - 1)
        return update * candidate + (1 - update) * deter

    def posterior(self, deter: jax.Array, token: jax.Array) -> jax.Array:
        """The logits [..., S, C] of q(z_t | h_t, e_t)."""
        x = self.posterior_hidden(jnp.concatenate([deter, token], -1))
        return self._reshape(self.posterior_logits(x))

    def prior(self, deter: jax.Array) -> jax.Array:
        """The logits [..., S, C] of p(z_t | h_t)."""
        return self._reshape(self.prior_logits(self.prior_hidden(deter)))

    def _reshape(self, logits: jax.Array) -> jax.Array:
        return logits.reshape(*logits.shape[:-1], self.stoch_size, self.classes)


def _expand(mask: jax.Array, x: jax.Array) -> jax.Array:
    """Adds trailing axes to a mask [B] so that it broadcasts against x."""
    return mask.reshape(mask.shape + (1,) * (x.ndim - mask.ndim)).astype(x.dtype)


def _swap_batch_and_time(tree):
    """Swaps axes 0 and 1. The model uses [B, T] arrays, but `jax.lax.scan`
    loops over axis 0, so the scans use [T, B] arrays."""
    return jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), tree)
