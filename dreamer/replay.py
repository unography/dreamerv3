"""Replay buffer.

The world model learns from sequences of consecutive steps, not from single
steps. Thus the replay buffer keeps the steps of each environment in time order
and gives sequences of a fixed length.

A sequence can contain the end of one episode and the start of the next one.
The `is_first` flag marks the start, and the world model resets its state at
that step.

Reference: appendix "Experience replay" of the DreamerV3 paper.
"""

import collections

import numpy as np


class Replay:
    """Keeps the recent steps of each environment and samples sequences.

    All environments add one step at the same time, in the same way that
    `BatchEnv` gives them. Each environment keeps its most recent
    `capacity // num_envs` steps. When the buffer is full, each new step
    replaces the oldest step of its environment.

    Each environment stream gives one sequence for each start position. The
    buffer samples all of these sequences with equal probability.

    With `online`, the buffer also keeps a queue of the new sequences. Each
    environment adds a sequence to the queue after each `length` new steps, so
    the queued sequences do not overlap. A batch takes the queued sequences
    first. Thus the agent trains on all new data at least once, and soon.
    """

    def __init__(
        self,
        capacity: int,
        num_envs: int,
        length: int,
        seed: int = 0,
        online: bool = False,
    ):
        self.num_envs = num_envs
        self.length = length
        self.slots = capacity // num_envs
        assert self.slots >= length, "The capacity is too small for one sequence."
        self.count = 0  # The number of steps that each environment added.
        self.data: dict[str, np.ndarray] = {}  # Arrays [num_envs, slots, ...].
        self.rng = np.random.default_rng(seed)
        self.online = online
        self.queue: collections.deque[tuple[int, int]] = collections.deque()

    def __len__(self) -> int:
        """The number of different sequences that the buffer can give."""
        stored = min(self.count, self.slots)
        return self.num_envs * max(0, stored - self.length + 1)

    def add(self, step: dict[str, np.ndarray]) -> None:
        """Adds one step [N, ...] that contains a value for each environment."""
        if not self.data:
            shape = (self.num_envs, self.slots)
            self.data = {
                key: np.zeros((*shape, *value.shape[1:]), value.dtype)
                for key, value in step.items()
            }
        slot = self.count % self.slots
        for key, value in step.items():
            self.data[key][:, slot] = value
        self.count += 1
        if self.online and self.count % self.length == 0:
            start = self.count - self.length
            self.queue.extend((env, start) for env in range(self.num_envs))

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Gives a batch of sequences [B, T, ...]."""
        assert len(self) > 0, "The buffer does not contain a full sequence."
        oldest = max(0, self.count - self.slots)
        queued = []
        while self.queue and len(queued) < batch_size:
            env, start = self.queue.popleft()
            if start >= oldest:  # The buffer still holds this sequence.
                queued.append((env, start))
        uniform = batch_size - len(queued)
        env = self.rng.integers(0, self.num_envs, uniform)
        start = self.rng.integers(oldest, self.count - self.length + 1, uniform)
        if queued:
            env = np.concatenate([[e for e, _ in queued], env])
            start = np.concatenate([[s for _, s in queued], start])
        slots = (start[:, None] + np.arange(self.length)) % self.slots
        batch = {key: value[env[:, None], slots] for key, value in self.data.items()}
        # The model does not know the state before the sequence. Thus it starts
        # each sequence from the reset state, as at the start of an episode.
        batch["is_first"][:, 0] = True
        return batch
