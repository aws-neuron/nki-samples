"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

Kernels written by the AWS Neuron.

WARNING: These kernels:
   - Are tested only against internal nightly builds
   - May not be compatible with public NeuronSDK releases
   - Have not been extensively tested across all input configurations
   - Carry no compatibility guarantees
   - The behavior of these kernels may be modified without prior notice

kernels - cumsum

"""
import numpy as np
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from common import div_ceil, normalize_dim, n_elts


def cumsum(x, y, axis=None, p_size=None, f_size=None, acc_dtype=None):
  '''
  Compute y = np.cumsum(x, axis=axis, dtype=dtype)

  :param x:
  :param y:
  :param axis:
  :param dtype:
  :return:
  '''

  assert isinstance(axis, int) or axis is None
  if axis is None:
    axis = -1

  rank = x.ndim

  axis = normalize_dim(axis, rank)
  assert axis == rank - 1, "Only support cusum over last dim"

  x_shape = x.shape
  shape_2d = (n_elts(x_shape[:-1]), x_shape[-1])
  x = x.reshape(shape_2d)

  assert y.shape == x_shape, "Expect x and y has the same shape!"

  pmax = nl.tile_size.pmax if p_size is None else p_size
  f_tile_size = 2048 if f_size is None else f_size

  pi, fi = nl.mgrid[0:pmax, 0:f_tile_size]

  acc_dtype = acc_dtype or x.dtype

  ones = nl.ones((pmax, f_tile_size), dtype=acc_dtype)
  # init = nl.zeros((pmax, 1), dtype=x.dtype)

  for i in nl.affine_range(div_ceil(shape_2d[0], pmax)):
    n_f_tiles = div_ceil(shape_2d[1], f_tile_size)
    init = nl.zeros((pmax, 1), dtype=acc_dtype)

    for j in nl.sequential_range(n_f_tiles):
      mask = (i * pmax + pi < shape_2d[0]) & (j * f_tile_size + fi < shape_2d[1])
      data = nl.load(x[i * pmax + pi, j * f_tile_size + fi], mask=mask)

      result = nisa.tensor_tensor_scan(data0=ones, data1=data, initial=init,
                                       op0=np.multiply, op1=np.add,
                                       dtype=acc_dtype, mask=mask)

      nl.store(y[i * pmax + pi, j * f_tile_size + fi], result, mask=mask)

      # update init for the next iteration
      init[:, :] = nl.copy(result[:, f_tile_size - 1], mask=j + 1 < n_f_tiles)
