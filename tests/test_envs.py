"""The step format of the environment wrapper."""

import numpy as np

from dreamer.envs import DMC


def test_reset_gives_the_first_step_of_an_episode():
    step = DMC("walker_walk").reset()

    assert step["is_first"]
    assert not step["is_last"]
    assert not step["is_terminal"]
    assert step["reward"] == 0


def test_observations_are_flat_float32_vectors():
    env = DMC("walker_walk")
    step = env.reset()

    assert env.obs_shapes == {"orientations": (14,), "height": (1,), "velocity": (9,)}
    for key, shape in env.obs_shapes.items():
        assert step[key].shape == shape
        assert step[key].dtype == np.float32


def test_walker_episodes_stop_at_a_time_limit_and_do_not_terminate():
    env = DMC("walker_walk")
    env.reset()

    steps = [env.step(np.zeros(env.action_size)) for _ in range(1000)]

    assert [s["is_last"] for s in steps] == [False] * 999 + [True]
    assert not any(s["is_terminal"] for s in steps)


def test_actions_outside_the_range_have_the_same_effect_as_the_bound():
    env_a, env_b = DMC("walker_walk", seed=1), DMC("walker_walk", seed=1)
    env_a.reset()
    env_b.reset()

    step_a = env_a.step(np.full(env_a.action_size, 5.0))
    step_b = env_b.step(np.full(env_b.action_size, 1.0))

    np.testing.assert_array_equal(step_a["velocity"], step_b["velocity"])
