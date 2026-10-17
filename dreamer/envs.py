"""Environments.

The agent sees each environment step as a dict of NumPy arrays:

- One float32 vector for each observation key, for example `velocity`.
- `reward`: the reward for the transition that arrives at this step.
- `is_first`: true for the first step of an episode. Its reward is zero.
- `is_last`: true for the last step of an episode.
- `is_terminal`: true if the episode ends because the task ends. It is false
  if the episode ends because of a time limit.

The difference between `is_last` and `is_terminal` is important. A terminal
step has no future rewards, so its value is zero. A walker episode always
stops at a time limit of 1000 steps. The world does not end there, so the agent
must not learn a value of zero for these steps.
"""

import os

import numpy as np

# The agent uses proprioceptive observations only. It does not render images,
# so dm_control does not need an OpenGL backend.
os.environ.setdefault("MUJOCO_GL", "disabled")

from dm_control import suite  # noqa: E402


class DMC:
    """A DeepMind Control Suite task with proprioceptive observations.

    The name has the form `domain_task`, for example `walker_walk`. The agent
    gives actions in [-1, 1]. The wrapper clips each action to this range and
    then scales it to the action bounds of the task.
    """

    def __init__(self, name: str = "walker_walk", seed: int = 0):
        domain, task = name.split("_", 1)
        self._env = suite.load(domain, task, task_kwargs={"random": seed})
        spec = self._env.action_spec()
        self._low, self._high = spec.minimum, spec.maximum

    @property
    def obs_shapes(self) -> dict[str, tuple[int]]:
        """The shape of each observation vector."""
        specs = self._env.observation_spec()
        return {key: (int(np.prod(spec.shape)),) for key, spec in specs.items()}

    @property
    def action_size(self) -> int:
        return int(self._env.action_spec().shape[0])

    def reset(self) -> dict[str, np.ndarray]:
        return self._convert(self._env.reset())

    def step(self, action: np.ndarray) -> dict[str, np.ndarray]:
        action = np.clip(action, -1, 1)
        action = self._low + (action + 1) / 2 * (self._high - self._low)
        return self._convert(self._env.step(action))

    def _convert(self, time_step) -> dict[str, np.ndarray]:
        first = time_step.first()
        obs = {
            key: np.asarray(value, np.float32).reshape(-1)
            for key, value in time_step.observation.items()
        }
        obs["reward"] = np.float32(0.0 if first else time_step.reward)
        obs["is_first"] = np.bool_(first)
        obs["is_last"] = np.bool_(time_step.last())
        obs["is_terminal"] = np.bool_(not first and time_step.discount == 0)
        return obs
