# Dreamer

This repository contains a small implementation of
[DreamerV3](https://arxiv.org/abs/2301.04104) in JAX. The neural networks use
[Flax NNX](https://flax.readthedocs.io/). The code supports one task family:
the DeepMind Control walker with proprioceptive observations.

The goal of this code is to teach. Each commit adds one part of the algorithm,
with tests that show what the part does. Read the commits in order.

## Install

Use Python 3.12 or later.

```sh
uv venv -p 3.12
uv pip install -e '.[test]'        # CPU only
uv pip install -e '.[cuda,test]'   # NVIDIA GPU
```

## Test

```sh
pytest
```

The tests run on the CPU. They use small models and finish in seconds.
