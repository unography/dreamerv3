import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from embodied.jax import nets
from embodied.jax import utils

f32 = jnp.float32


class Cell(nj.Module):

  units: int = 16
  tapgrad: bool = True

  def __call__(self, h, xs):
    scan = utils.tap_scan if self.tapgrad else nj.scan
    return scan(self.step, h, xs, axis=1)

  def step(self, h, inp):
    x, reset = inp
    h = nets.mask(h, ~reset)
    x = self.sub('inp', nets.Linear, self.units)(x)
    x = jax.nn.silu(self.sub('inpnorm', nets.Norm, 'rms')(x))
    x = jnp.concatenate([h, x], -1)
    x = self.sub('hid', nets.BlockLinear, 2 * self.units, 4)(x)
    x = jax.nn.silu(self.sub('hidnorm', nets.Norm, 'rms')(x))
    noise = jax.random.normal(nj.seed(), x.shape, x.dtype)
    x = self.sub('out', nets.Linear, self.units, bias=False)(x + 0.1 * noise)
    h = jnp.tanh(x) + 0.5 * h
    return h, h

  def loss(self, h, xs):
    h, ys = self(h, xs)
    return jnp.square(ys - 0.3).mean() + h.mean()


class TestTapScan:

  def setup_method(self):
    self.dtype, nets.COMPUTE_DTYPE = nets.COMPUTE_DTYPE, f32

  def teardown_method(self):
    nets.COMPUTE_DTYPE = self.dtype

  def _setup(self, B=3, T=5, D=6, U=16):
    rng = np.random.default_rng(0)
    h = jnp.asarray(rng.normal(0, 1, (B, U)), f32)
    x = jnp.asarray(rng.normal(0, 1, (B, T, D)), f32)
    reset = jnp.asarray(rng.random((B, T)) < 0.3)
    params = nj.init(Cell(units=U, name='cell').loss)(
        {}, h, (x, reset), seed=0)
    return params, h, (x, reset)

  def _grads(self, tapgrad, params, h, xs):
    cell = Cell(units=h.shape[-1], tapgrad=tapgrad, name='cell')
    def fn(h, xs):
      loss, _, grads = nj.grad(cell.loss, [cell])(h, xs)
      return loss, grads
    fn = nj.pure(fn)
    jaxpr = str(jax.make_jaxpr(lambda *a: fn(*a, seed=1))(params, h, xs))
    with jax.default_matmul_precision('highest'):
      _, (loss, grads) = jax.jit(fn)(params, h, xs, seed=1)
    return loss, grads, jaxpr.count('custom_vjp_call')

  def test_forward(self):
    params, h, xs = self._setup()
    outs = []
    for tapgrad in (False, True):
      cell = Cell(units=h.shape[-1], tapgrad=tapgrad, name='cell')
      outs.append(nj.pure(cell)(params, h, xs, seed=1)[1])
    for a, b in zip(jax.tree.leaves(outs[0]), jax.tree.leaves(outs[1])):
      assert np.allclose(a, b, rtol=0, atol=0)

  def test_grads(self):
    params, h, xs = self._setup()
    loss1, grads1, count1 = self._grads(False, params, h, xs)
    loss2, grads2, count2 = self._grads(True, params, h, xs)
    assert count2 > count1
    assert np.allclose(loss1, loss2, rtol=1e-6)
    assert grads1.keys() == grads2.keys() == params.keys()
    for key in grads1:
      assert np.abs(grads1[key]).max() > 0, key
      assert np.allclose(grads1[key], grads2[key], rtol=1e-5, atol=1e-6), key

  def test_input_grads(self):
    params, h, xs = self._setup()
    def fn(tapgrad):
      cell = Cell(units=h.shape[-1], tapgrad=tapgrad, name='cell')
      loss = lambda h, x: nj.pure(cell.loss)(
          params, h, (x, xs[1]), seed=1)[1]
      with jax.default_matmul_precision('highest'):
        return jax.grad(loss, (0, 1))(h, xs[0])
    for a, b in zip(fn(False), fn(True)):
      assert np.allclose(a, b, rtol=1e-5, atol=1e-6)
