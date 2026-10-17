"""Trains the agent on a DMC task.

The training loop does two things in turns:

- The agent acts in N environments, and the replay buffer keeps the steps.
- The agent trains on sequences from the replay buffer.

The train ratio is the number of replayed steps for each environment step. A
batch has 16 x 64 = 1024 steps, so a train ratio of 1024 gives one training
step for each environment step.

Usage:

    python -m dreamer.train --task walker_walk --logdir ~/logdir/walker
    python -m dreamer.train --help
"""

import dataclasses
import json
import pathlib
import time

import jax
import numpy as np
import tyro
from flax import nnx

from dreamer.agent import Agent
from dreamer.config import Config
from dreamer.envs import DMC, BatchEnv
from dreamer.replay import Replay


@dataclasses.dataclass(frozen=True)
class Args:
    task: str = "walker_walk"
    logdir: str = "~/logdir/dreamer"
    seed: int = 0
    steps: int = 1_100_000  # The number of environment steps.
    envs: int = 16
    train_ratio: float = 1024
    batch_size: int = 16
    batch_length: int = 64
    replay_size: int = 5_000_000
    log_every: int = 10_000  # The number of environment steps between logs.
    agent: Config = Config()


class Ratio:
    """Gives the number of training steps to do after each environment step.

    The total number of training steps grows at `ratio` times the number of
    environment steps. The count starts at the first call, so the steps
    before the start of training do not count.
    """

    def __init__(self, ratio: float):
        self.ratio = ratio
        self.last = None

    def __call__(self, step: int) -> int:
        if self.last is None:
            self.last = step
            return 1
        repeats = int((step - self.last) * self.ratio)
        self.last += repeats / self.ratio
        return repeats


class Episodes:
    """Keeps the return and the length of the episode in each environment."""

    def __init__(self, num_envs: int):
        self.returns = np.zeros(num_envs)
        self.lengths = np.zeros(num_envs, int)
        self.finished: list[tuple[float, int]] = []

    def add(self, obs: dict[str, np.ndarray]) -> None:
        self.returns = np.where(obs["is_first"], 0, self.returns) + obs["reward"]
        self.lengths = np.where(obs["is_first"], 0, self.lengths) + 1
        for env in np.flatnonzero(obs["is_last"]):
            self.finished.append((self.returns[env], self.lengths[env]))

    def summary(self) -> dict[str, float]:
        """The mean return and length of the episodes since the last summary."""
        if not self.finished:
            return {}
        returns, lengths = zip(*self.finished, strict=True)
        self.finished = []
        return {"episode/score": np.mean(returns), "episode/length": np.mean(lengths)}


def mean_metrics(metrics: list[dict]) -> dict[str, float]:
    """The mean of each metric over a list of training steps."""
    metrics = jax.device_get(metrics)
    return {
        f"train/{name}": float(np.mean([m[name] for m in metrics]))
        for name in (metrics[0] if metrics else {})
    }


class Logger:
    """Writes metrics to `metrics.jsonl` in the log directory and prints them."""

    def __init__(self, logdir: pathlib.Path):
        self.path = logdir / "metrics.jsonl"

    def write(self, step: int, metrics: dict[str, float]) -> None:
        metrics = {name: float(value) for name, value in metrics.items()}
        with self.path.open("a") as file:
            file.write(json.dumps({"step": step, **metrics}) + "\n")
        shown = ("episode/score", "train/loss/rec", "train/loss/policy", "fps/env")
        items = [f"{name} {metrics[name]:.4g}" for name in shown if name in metrics]
        print(f"step {step}", *items, sep=" | ", flush=True)


def train(args: Args) -> None:
    logdir = pathlib.Path(args.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / "args.json").write_text(json.dumps(dataclasses.asdict(args), indent=2))
    logger = Logger(logdir)

    envs = BatchEnv(
        [DMC(args.task, seed=args.seed * args.envs + i) for i in range(args.envs)]
    )
    agent = Agent(
        envs.obs_shapes, envs.action_size, args.agent, rngs=nnx.Rngs(args.seed)
    )
    replay = Replay(
        min(args.replay_size, args.steps), args.envs, args.batch_length, args.seed
    )
    batch_steps = args.batch_size * args.batch_length
    should_train = Ratio(args.train_ratio / batch_steps)
    episodes = Episodes(args.envs)
    key = jax.random.key(args.seed)

    policy_state = agent.initial_policy_state(args.envs)
    action = np.zeros((args.envs, envs.action_size), np.float32)
    train_metrics = []
    step = last_log = 0
    last_time = time.time()

    while step < args.steps:
        # Act in the environments.
        obs = envs.step(action)
        key, policy_key = jax.random.split(key)
        action, policy_state = agent.policy(policy_state, obs, policy_key)
        action = np.asarray(action)
        replay.add({**obs, "action": action})  # The action after the observation.
        episodes.add(obs)
        step += args.envs

        # Train on replay data, after the buffer has enough sequences.
        if len(replay) >= batch_steps:
            for _ in range(should_train(step)):
                key, train_key = jax.random.split(key)
                batch = replay.sample(args.batch_size)
                train_metrics.append(agent.train(batch, train_key))

        if step - last_log >= args.log_every:
            fps = (step - last_log) / (time.time() - last_time)
            logger.write(
                step,
                {
                    "fps/env": fps,
                    "train_steps": agent.optimizer.step[...],
                    **episodes.summary(),
                    **mean_metrics(train_metrics),
                },
            )
            train_metrics = []
            last_log, last_time = step, time.time()


if __name__ == "__main__":
    train(tyro.cli(Args))
