"""Plots the episode scores of training runs.

Usage:

    python scripts/plot.py run1/metrics.jsonl run2/metrics.jsonl --out scores.png

Each input is a JSONL file with rows that have the keys `step` and
`episode/score`. The `metrics.jsonl` file of `dreamer.train` and the
`scores.jsonl` file of the reference implementation have this format. The plot
shows the mean score in bins of environment steps.
"""

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np


def load_scores(path: pathlib.Path, bin_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Gives the end step and the mean score of each bin of steps."""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows = [row for row in rows if "episode/score" in row]
    steps = np.array([row["step"] for row in rows])
    scores = np.array([row["episode/score"] for row in rows])
    bins = (steps - 1) // bin_size
    ends = (np.unique(bins) + 1) * bin_size
    means = np.array([scores[bins == b].mean() for b in np.unique(bins)])
    return ends, means


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", type=pathlib.Path)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--bin-size", type=int, default=16_016)
    parser.add_argument("--title", default="walker_walk")
    parser.add_argument("--out", type=pathlib.Path, default="scores.png")
    args = parser.parse_args()

    labels = args.labels or [str(file.parent.name) for file in args.files]
    fig, ax = plt.subplots(figsize=(6, 4))
    for file, label in zip(args.files, labels, strict=True):
        ax.plot(*load_scores(file, args.bin_size), marker="o", label=label)
    ax.set_title(args.title)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode score")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
