"""What the replay buffer keeps and how it gives sequences."""

import numpy as np

from dreamer.replay import Location, Replay


def add_steps(replay, num_steps, episode_length=1000):
    """Adds steps that record their environment and time.

    The value `id` of a step is `1000 * env + time`. Thus a test can find the
    source of each sampled step.
    """
    for _ in range(num_steps):
        time = replay.count
        replay.add(
            {
                "id": np.array([1000 * env + time for env in range(replay.num_envs)]),
                "is_first": np.full(replay.num_envs, time % episode_length == 0),
            }
        )


def test_buffer_gives_no_sequences_until_one_sequence_fits():
    replay = Replay(capacity=100, num_envs=2, length=5)

    add_steps(replay, 4)
    assert len(replay) == 0

    add_steps(replay, 1)
    assert len(replay) == 2

    add_steps(replay, 3)
    assert len(replay) == 2 * 4


def test_sequences_are_consecutive_steps_of_one_environment():
    replay = Replay(capacity=100, num_envs=3, length=5)
    add_steps(replay, 20)

    ids = replay.sample(batch_size=64)[0]["id"]

    env, time = ids // 1000, ids % 1000
    assert (env == env[:, :1]).all()
    assert (np.diff(time, axis=1) == 1).all()


def test_sampling_reaches_every_sequence():
    replay = Replay(capacity=100, num_envs=2, length=5)
    add_steps(replay, 10)

    starts = {int(seq[0]) for seq in replay.sample(batch_size=1000)[0]["id"]}

    assert starts == {1000 * env + time for env in range(2) for time in range(6)}


def test_full_buffer_forgets_the_oldest_steps():
    replay = Replay(capacity=20, num_envs=2, length=5)

    add_steps(replay, 25)

    times = replay.sample(batch_size=1000)[0]["id"] % 1000
    assert times.min() == 15
    assert times.max() == 24
    assert len(replay) == 2 * 6


def test_each_sequence_starts_with_is_first():
    replay = Replay(capacity=100, num_envs=1, length=5)
    add_steps(replay, 30, episode_length=7)

    batch, _ = replay.sample(batch_size=200)

    assert batch["is_first"][:, 0].all()
    # The other steps keep their flags, so episode starts inside a sequence
    # stay visible.
    expected = batch["id"][:, 1:] % 7 == 0
    np.testing.assert_array_equal(batch["is_first"][:, 1:], expected)
    # The buffer itself does not change.
    assert replay.data["is_first"].sum() == 5


def test_online_queue_gives_each_new_sequence_first():
    replay = Replay(capacity=100, num_envs=2, length=5, online=True)
    add_steps(replay, 10)

    first = replay.sample(batch_size=3)[0]["id"][:, 0]
    second = replay.sample(batch_size=3)[0]["id"][:, 0]

    # Each environment queued the sequences that start at time 0 and 5.
    assert first.tolist() == [0, 1000, 5]
    assert second[0] == 1005


def test_online_queue_gives_every_new_step_to_training():
    replay = Replay(capacity=1000, num_envs=2, length=5, online=True)
    seen = set()
    for _ in range(20):
        add_steps(replay, 5)
        seen.update(replay.sample(batch_size=4)[0]["id"].flatten().tolist())

    all_steps = {1000 * env + time for env in range(2) for time in range(100)}
    assert seen == all_steps


def test_update_writes_new_values_into_the_last_steps_of_the_sequences():
    replay = Replay(capacity=100, num_envs=2, length=5)
    add_steps(replay, 10)
    location = Location(env=np.array([0, 1]), start=np.array([1, 3]))

    replay.update(location, {"id": np.full((2, 4), -1)})

    assert replay.data["id"][0, :10].tolist() == [0, 1, -1, -1, -1, -1, 6, 7, 8, 9]
    assert replay.data["id"][1, :10].tolist() == (
        [1000, 1001, 1002, 1003, -1, -1, -1, -1, 1008, 1009]
    )
