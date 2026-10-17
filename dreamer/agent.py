"""The agent: the world model, the actor, the critic, and their training.

One training step does these operations on a batch of replay sequences:

1. The world model observes the sequences and gives its losses, and the
   posterior state of each step.
2. The actor and the critic imagine trajectories from all of these states and
   give their losses.
3. One optimizer step on the sum of all losses trains all networks. Stop
   gradients keep the losses apart: the world model learns only from its own
   losses.
4. The slow critic moves toward the critic.

In the environment, the agent acts with `policy`. The policy keeps the RSSM
state of each environment, updates it with each observation, and samples an
action from the actor.
"""

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from dreamer.actor_critic import (
    Actor,
    ReturnNorm,
    imagination_loss,
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
            rngs=rngs,
        )
        self.critic = TwoHotHead(feature_size, units=c.units, layers=3, rngs=rngs)
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
    def train(self, batch: dict[str, jax.Array], key: jax.Array) -> dict:
        """Does one training step on a batch of sequences [B, T, ...].

        Returns:
            The metrics of the step.
        """
        grad_fn = nnx.grad(
            Agent.loss, argnums=nnx.DiffState(0, TRAINABLE), has_aux=True
        )
        grads, metrics = grad_fn(self, batch, key)
        updates = self.optimizer.update(self, grads)
        update_slow_critic(self.slow_critic, self.critic, self.config.slow_rate)
        metrics["grad_norm"] = optax.tree.norm(grads)
        metrics["update_norm"] = optax.tree.norm(updates)
        return metrics

    def loss(
        self, batch: dict[str, jax.Array], key: jax.Array
    ) -> tuple[jax.Array, dict]:
        """Gives the sum of all scaled losses, and the metrics."""
        c = self.config
        wm_key, imagine_key = jax.random.split(key)
        B, T = batch["is_first"].shape

        # Each step of the replay data holds the action after its observation.
        # The world model needs the action before each observation.
        actions = batch["action"]
        prev_actions = jnp.concatenate(
            [jnp.zeros_like(actions[:, :1]), actions[:, :-1]], 1
        )
        start = self.world_model.rssm.initial(B)
        wm_losses, states = self.world_model.loss(
            start, prev_actions, batch, wm_key, horizon=c.horizon, free_nats=c.free_nats
        )

        # The imagination starts from the posterior state of each replay step.
        starts = jax.tree.map(lambda x: x.reshape(B * T, *x.shape[2:]), states)
        ac_losses, _, ac_metrics = imagination_loss(
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

        losses = {**wm_losses, **ac_losses}
        scales = {
            "dyn": c.dyn_scale,
            "rep": c.rep_scale,
            "rec": c.rec_scale,
            "rew": c.rew_scale,
            "con": c.con_scale,
            "policy": c.policy_scale,
            "value": c.value_scale,
        }
        total = sum(scales[name] * loss.mean() for name, loss in losses.items())
        metrics = {f"loss/{name}": loss.mean() for name, loss in losses.items()}
        metrics.update(ac_metrics)
        return total, metrics
