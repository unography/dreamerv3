"""The training loop: the train ratio, the episode statistics, and a full run."""

import json

import numpy as np

from dreamer.config import Config
from dreamer.train import Args, Episodes, Ratio, train


def test_train_ratio_of_one_gives_one_training_step_per_environment_step():
    # With 16 environments, each loop iteration adds 16 environment steps.
    ratio = Ratio(1024 / (16 * 64))

    counts = [ratio(step) for step in range(0, 16 * 5, 16)]

    assert counts == [1, 16, 16, 16, 16]


def test_small_train_ratio_trains_after_some_environment_steps():
    ratio = Ratio(0.25)

    counts = [ratio(step) for step in range(1, 101)]

    assert sum(counts) == 1 + 24
    assert set(counts) == {0, 1}


def test_episodes_give_the_return_and_length_of_finished_episodes():
    episodes = Episodes(num_envs=2)
    steps = [
        # Rewards of env 0 and env 1, is_first, is_last.
        ([0, 0], [True, True], [False, False]),
        ([1, 2], [False, False], [False, True]),
        ([3, 0], [False, True], [True, False]),
    ]
    ended = [
        episodes.add(
            {
                "reward": np.array(reward, np.float32),
                "is_first": np.array(is_first),
                "is_last": np.array(is_last),
            }
        )
        for reward, is_first, is_last in steps
    ]

    assert ended == [[], [(2.0, 2)], [(4.0, 3)]]
    summary = episodes.summary()
    assert summary == {"episode/score": (2 + 4) / 2, "episode/length": (2 + 3) / 2}
    assert episodes.summary() == {}


def test_training_runs_and_writes_metrics(tmp_path):
    args = Args(
        task="walker_walk",
        logdir=str(tmp_path),
        envs=2,
        steps=2100,  # A walker episode has 1001 steps, with the first step.
        train_ratio=1,
        batch_size=2,
        batch_length=8,
        log_every=700,
        agent=Config(deter=16, hidden=8, stoch=4, classes=3, blocks=2, units=8),
    )

    train(args)

    lines = (tmp_path / "metrics.jsonl").read_text().splitlines()
    metrics = [json.loads(line) for line in lines]
    assert [m["step"] for m in metrics] == [700, 1400, 2100]
    assert metrics[-1]["train_steps"] > 0
    assert np.isfinite(metrics[-1]["train/loss/rec"])
    assert 0 <= metrics[-1]["episode/score"] <= 1000

    lines = (tmp_path / "scores.jsonl").read_text().splitlines()
    scores = [json.loads(line) for line in lines]
    assert [row["step"] for row in scores] == [2002, 2002]
    assert [row["episode/length"] for row in scores] == [1001, 1001]
