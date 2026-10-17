"""How the batch runner starts and resets episodes."""

import numpy as np

from dreamer.envs import BatchEnv


class CountingEnv:
    """An environment with episodes of a fixed length.

    The observation is the step count in the episode. The reward is equal to
    the action, so the tests can see which actions have an effect.
    """

    obs_shapes = {"count": (1,)}
    action_size = 1

    def __init__(self, length: int):
        self.length = length
        self.count = 0

    def reset(self):
        self.count = 0
        return self._step(reward=0.0)

    def step(self, action):
        self.count += 1
        return self._step(reward=float(action[0]))

    def _step(self, reward):
        return {
            "count": np.array([self.count], np.float32),
            "reward": np.float32(reward),
            "is_first": np.bool_(self.count == 0),
            "is_last": np.bool_(self.count == self.length),
            "is_terminal": np.bool_(False),
        }


def run(batch_env, num_steps, action=1.0):
    actions = np.full((len(batch_env), 1), action, np.float32)
    return [batch_env.step(actions) for _ in range(num_steps)]


def test_first_call_starts_an_episode_in_every_environment():
    batch_env = BatchEnv([CountingEnv(3), CountingEnv(3)])

    (step,) = run(batch_env, 1)

    assert step["is_first"].tolist() == [True, True]
    assert step["reward"].tolist() == [0, 0]


def test_step_after_the_last_step_starts_a_new_episode():
    batch_env = BatchEnv([CountingEnv(3)])

    steps = run(batch_env, 9)

    counts = [int(step["count"][0, 0]) for step in steps]
    firsts = [bool(step["is_first"][0]) for step in steps]
    lasts = [bool(step["is_last"][0]) for step in steps]
    assert counts == [0, 1, 2, 3, 0, 1, 2, 3, 0]
    assert firsts == [True, False, False, False, True, False, False, False, True]
    assert lasts == [False, False, False, True, False, False, False, True, False]


def test_action_after_the_last_step_has_no_effect():
    batch_env = BatchEnv([CountingEnv(1)])

    steps = run(batch_env, 3, action=5.0)

    assert [float(step["reward"][0]) for step in steps] == [0.0, 5.0, 0.0]


def test_environments_reset_independently():
    batch_env = BatchEnv([CountingEnv(1), CountingEnv(2)])

    steps = run(batch_env, 4)

    assert [step["is_first"].tolist() for step in steps] == [
        [True, True],
        [False, False],
        [True, False],
        [False, True],
    ]
