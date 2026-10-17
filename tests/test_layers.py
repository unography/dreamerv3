"""The behavior of the network layers."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from dreamer.layers import MLP, BlockLinear, scaled_init


def test_block_linear_is_a_dense_layer_with_a_block_diagonal_matrix():
    layer = BlockLinear(12, 6, blocks=3, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(1), (5, 12))

    matrix = jax.scipy.linalg.block_diag(*layer.kernel[...])

    assert matrix.shape == (12, 6)
    np.testing.assert_allclose(layer(x), x @ matrix + layer.bias[...], atol=1e-6)


def test_block_linear_output_group_depends_only_on_its_input_group():
    layer = BlockLinear(12, 6, blocks=3, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(1), (12,))

    changed = layer(x.at[4:8].add(1.0)) != layer(x)

    assert changed.tolist() == [False, False, True, True, False, False]


def test_layer_with_zero_scale_outputs_zero():
    layer = nnx.Linear(8, 4, kernel_init=scaled_init(0.0), rngs=nnx.Rngs(0))

    np.testing.assert_array_equal(layer(jnp.ones((2, 8))), 0.0)


def test_default_initialization_keeps_the_activation_scale():
    layer = nnx.Linear(1024, 1024, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(1), (256, 1024))

    np.testing.assert_allclose(layer(x).std(), 1.0, atol=0.05)


def test_mlp_output_does_not_depend_on_the_input_scale():
    # Each layer normalizes its activations, so a larger input does not give
    # larger activations. Only the bias of the first layer breaks this rule,
    # and it is zero at the start.
    mlp = MLP(16, units=32, layers=3, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(1), (4, 16))

    np.testing.assert_allclose(mlp(1000 * x), mlp(x), atol=1e-3)
