"""The agent: the world model, the actor, the critic, and their training.

One training step does these operations on a batch of replay sequences:

1. The world model observes the sequences and gives its losses, and the
   posterior state of each step.
2. The actor and the critic imagine trajectories from all of these states and
   give their losses.
3. The critic also learns the returns of the replay sequences.
4. One optimizer step on the sum of all losses trains all networks. Stop
   gradients keep the losses apart: the actor-critic losses on imagined
   trajectories do not train the world model.
5. The slow critic moves toward the critic.

In the environment, the agent acts with `policy`. The policy keeps the RSSM
state of each environment, updates it with each observation, and samples an
action from the actor.

Replay context: the replay buffer also keeps the RSSM state of each step,
first from the policy and then from each training step on the step. Thus a
training sequence can start from the state of its first step, instead of from
zero. See `Agent.split_context`.
"""

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from dreamer.actor_critic import (
    Actor,
    ReturnNorm,
    imagination_loss,
    replay_value_loss,
    update_slow_critic,
)
from dreamer.config import Config
from dreamer.optim import make_optimizer
from dreamer.rssm import State
from dreamer.world_model import TwoHotHead, WorldModel

# The state of the policy: the RSSM state and the last action [N, A].
PolicyState = tuple[State, jax.Array]

# The optimizer trains all weights except the weights of the slow critic, which
# follows the critic.
TRAINABLE = nnx.All(nnx.Param, nnx.Not(nnx.PathContains("slow_critic")))


class Agent(nnx.Module):
    def __init__(
        self,
        obs_shapes: dict[str, tuple[int]],
        action_size: int,
        config: Config,
        *,
        rngs: nnx.Rngs,
    ):
        c = self.config = config
        self.action_size = action_size
        dtype = jnp.dtype(c.compute_dtype)
        self.world_model = WorldModel(
            obs_shapes,
            action_size,
            deter=c.deter,
            hidden=c.hidden,
            stoch=c.stoch,
            classes=c.classes,
            blocks=c.blocks,
            units=c.units,
            unimix=c.unimix,
            dtype=dtype,
            rngs=rngs,
        )
        feature_size = c.deter + c.stoch * c.classes
        self.actor = Actor(
            feature_size,
            action_size,
            units=c.units,
            layers=3,
            min_std=c.min_std,
            max_std=c.max_std,
            dtype=dtype,
            rngs=rngs,
        )
        self.critic = TwoHotHead(
            feature_size, units=c.units, layers=3, dtype=dtype, rngs=rngs
        )
        self.slow_critic = nnx.clone(self.critic)
        self.return_norm = ReturnNorm(rate=c.return_norm_rate)
        self.optimizer = nnx.Optimizer(
            self,
            make_optimizer(c.lr, c.agc, c.eps, c.beta1, c.beta2, c.warmup),
            wrt=TRAINABLE,
        )

    def initial_policy_state(self, num_envs: int) -> PolicyState:
        initial = self.world_model.rssm.initial(num_envs)
        return initial, jnp.zeros((num_envs, self.action_size))

    @nnx.jit(graph=False)
    def policy(
        self, state: PolicyState, obs: dict[str, jax.Array], key: jax.Array
    ) -> tuple[jax.Array, PolicyState]:
        """Chooses the actions [N, A] for one step of N environments.

        The world model updates its state with the new observation. Then the
        actor samples an action from this state. The samples are random, so
        the agent explores.
        """
        rssm_state, prev_action = state
        obs_key, action_key = jax.random.split(key)
        world_model = self.world_model
        token = world_model.encoder({name: obs[name] for name in world_model.obs_keys})
        rssm_state, _ = world_model.rssm.obs_step(
            rssm_state, prev_action, token, obs["is_first"], obs_key
        )
        action = self.actor(rssm_state.features()).sample(action_key)
        return action, (rssm_state, action)

    # In tree mode (graph=False), `nnx.jit` handles the agent as a plain pytree.
    # The default graph mode supports shared variables, but then each call
    # walks the graph of the agent in Python, which takes longer than a policy
    # step on the GPU.
    @nnx.jit(graph=False)
    def train(self, batch: dict[str, jax.Array], key: jax.Array) -> tuple[dict, State]:
        """Does one training step on a batch of sequences [B, T, ...].

        Returns:
            The metrics of the step, and the posterior states of the trained
            steps. With the replay context, these are the last T - 1 steps.
            The replay buffer keeps these states for the replay context.
        """
        grad_fn = nnx.grad(
            Agent.loss, argnums=nnx.DiffState(0, TRAINABLE), has_aux=True
        )
        grads, (metrics, states) = grad_fn(self, batch, key)
        updates = self.optimizer.update(self, grads)
        update_slow_critic(self.slow_critic, self.critic, self.config.slow_rate)
        metrics["grad_norm"] = optax.tree.norm(grads)
        metrics["update_norm"] = optax.tree.norm(updates)
        return metrics, states

    def loss(
        self, batch: dict[str, jax.Array], key: jax.Array
    ) -> tuple[jax.Array, tuple[dict, State]]:
        """Gives the sum of all scaled losses, the metrics, and the states."""
        c = self.config
        wm_key, imagine_key = jax.random.split(key)
        start, prev_actions, batch = self.split_context(batch)
        B, T = batch["is_first"].shape

        wm_losses, states = self.world_model.loss(
            start, prev_actions, batch, wm_key, horizon=c.horizon, free_nats=c.free_nats
        )

        # The imagination starts from the posterior state of each replay step.
        starts = jax.tree.map(lambda x: x.reshape(B * T, *x.shape[2:]), states)
        ac_losses, imagined_returns, ac_metrics = imagination_loss(
            self.world_model,
            self.actor,
            self.critic,
            self.slow_critic,
            self.return_norm,
            starts,
            imagine_key,
            horizon=c.imag_horizon,
            lam=c.lam,
            entropy_scale=c.entropy_scale,
            slow_reg=c.slow_reg,
        )

        # The critic also learns the returns of the replay sequences. The
        # imagined return from each replay step is the bootstrap value.
        repval_loss, _ = replay_value_loss(
            self.critic,
            self.slow_critic,
            states.features(),
            batch["reward"],
            batch["is_last"],
            batch["is_terminal"],
            imagined_returns[:, 0].reshape(B, T),
            horizon=c.horizon,
            lam=c.lam,
            slow_reg=c.slow_reg,
        )

        losses = {**wm_losses, **ac_losses, "repval": repval_loss}
        scales = {
            "dyn": c.dyn_scale,
            "rep": c.rep_scale,
            "rec": c.rec_scale,
            "rew": c.rew_scale,
            "con": c.con_scale,
            "policy": c.policy_scale,
            "value": c.value_scale,
            "repval": c.repval_scale,
        }
        total = sum(scales[name] * loss.mean() for name, loss in losses.items())
        metrics = {f"loss/{name}": loss.mean() for name, loss in losses.items()}
        metrics.update(ac_metrics)
        return total, (metrics, states)

    def split_context(
        self, batch: dict[str, jax.Array]
    ) -> tuple[State, jax.Array, dict[str, jax.Array]]:
        """Splits a replay batch into a start state, actions, and steps.

        Each replay step holds the action after its observation. The world
        model needs the action before each observation, so the actions move
        by one step.

        With the replay context, the first step of each sequence only gives
        the start state: the state that the replay buffer keeps for this step.
        Thus the training continues from the state of the last training or
        the policy, and does not start the sequence from zero. Without the
        replay context, each sequence starts from the initial state.

        Returns:
            The start state [B, ...], the previous actions [B, T', A], and the
            steps [B, T', ...] to train on.
        """
        actions = batch["action"]
        if self.config.replay_context:
            start = State(batch["deter"][:, 0], batch["stoch"][:, 0])
            steps = {key: value[:, 1:] for key, value in batch.items()}
            return start, actions[:, :-1], steps
        start = self.world_model.rssm.initial(actions.shape[0])
        first_action = jnp.zeros_like(actions[:, :1])
        prev_actions = jnp.concatenate([first_action, actions[:, :-1]], 1)
        return start, prev_actions, batch
