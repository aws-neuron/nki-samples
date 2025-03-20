"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

Kernels written by the AWS Neuron.

WARNING: These kernels:
   - Are tested only against internal nightly builds
   - May not be compatible with public NeuronSDK releases
   - Have not been extensively tested across all input configurations
   - Carry no compatibility guarantees
   - The behavior of these kernels may be modified without prior notice

Helpers functions that can be called as part of other top level kernels.

"""
import numpy as np

import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc.nki.language.iterators import sync_program
from neuronxcc.nki.language import par_dim
from common import n_elts


def softmax_kernel(src, compute_dtype, tile_reduce_indices,
                   program_reduce_axes, scale, mask):
  if scale < 0:
    src = nl.multiply(src, scale, dtype=compute_dtype, mask=mask)
    scale = 1.0

  if src.buffer is nl.psum:
    # Move the data from PSUM to SB first
    src = nl.copy(src, dtype=compute_dtype, mask=mask)

  assert scale > 0, "unexpected scale"
  exp_ = _softmax_exp(src, compute_dtype=compute_dtype, scale=scale,
                      tile_reduce_indices=tile_reduce_indices,
                      program_reduce_axes=program_reduce_axes, mask=mask)

  exp_sum = exp_
  if tile_reduce_indices:
    exp_sum = nisa.tensor_reduce(np.add, data=exp_, axis=tile_reduce_indices,
                          dtype=compute_dtype, mask=mask)

  # Reduce between kernels
  if program_reduce_axes:
    exp_sum = nl.all_reduce(exp_sum, op=np.add, dtype=compute_dtype,
                            program_axes=program_reduce_axes,
                            mask=mask)

  exp_sum = exp_sum.expand_dims(tuple(range(exp_sum.ndim, exp_.ndim)))

  return nl.divide(exp_, exp_sum, mask=mask)


def softmax_exp_kernel(src, compute_dtype, tile_reduce_indices,
                       program_reduce_axes, scale, mask):
  return _softmax_exp(src, compute_dtype=compute_dtype, scale=scale,
                      tile_reduce_indices=tile_reduce_indices,
                      program_reduce_axes=program_reduce_axes, mask=mask)


def softmax_rsum_kernel(src, compute_dtype, tile_reduce_indices,
                        program_reduce_axes, mask):
  exp_sum = src
  if tile_reduce_indices:
    exp_sum = nisa.tensor_reduce(np.add, data=src, axis=tile_reduce_indices,
                                 dtype=compute_dtype, mask=mask)

  # Reduce between kernels
  if program_reduce_axes:
    exp_sum = nl.all_reduce(exp_sum, op=np.add, dtype=compute_dtype,
                            program_axes=program_reduce_axes,
                            mask=mask)

  # Assign the computed softmax rsum result to dst
  return nl.divide(1, exp_sum, mask=mask)


def _softmax_exp(src, compute_dtype, tile_reduce_indices, program_reduce_axes,
                 scale, mask):

  # Cast scale if integer
  scale = float(scale) if isinstance(scale, int) else scale

  # Reduce inside the tile
  if tile_reduce_indices:
    negate_src_max = nisa.tensor_reduce(np.max, data=src,
                                 axis=tile_reduce_indices,
                                 dtype=compute_dtype, negate=True, mask=mask)
  else:
    negate_src_max = nl.multiply(src, -1, dtype=compute_dtype, mask=mask)

  # Reduce between spmd kernels on the same device
  if program_reduce_axes:
    negate_src_max = nl.all_reduce(negate_src_max, op=np.min, dtype=compute_dtype,
                                   program_axes=program_reduce_axes,
                                   mask=mask
                                   )
  # Use bias of the lowlevel activation API if possible
  if negate_src_max.shape_2d[1] == 1:
    if scale != 1.0:
      bias = nl.multiply(negate_src_max, scale, dtype=compute_dtype, mask=mask)
    else:
      bias = negate_src_max
    exp_ = nisa.activation(np.exp, data=src, bias=bias, dtype=compute_dtype,
                           scale=scale, mask=mask)
  else:
    negate_src_max = negate_src_max.expand_dims(tuple(range(negate_src_max.ndim, src.ndim)))
    data = nl.add(src, negate_src_max, dtype=compute_dtype, mask=mask)
    exp_ = nisa.activation(np.exp, data=data, scale=scale, mask=mask)
  return exp_


def softmax_dx_kernel(grad_out, fwd_out, compute_dtype,
                      tile_reduce_indices, program_reduce_axes, mask):
  if fwd_out.buffer is nl.psum:
    # Move the data from PSUM to SB first
    fwd_out = nl.copy(fwd_out, dtype=compute_dtype, mask=mask)

  _multiply = nl.multiply(grad_out, fwd_out, dtype=compute_dtype, mask=mask)
  # Reduce inside the tile
  _sum = _multiply
  if tile_reduce_indices:
    _sum = nisa.tensor_reduce(np.add, data=_multiply,
                       axis=tile_reduce_indices,
                       dtype=compute_dtype, mask=mask)

  # FIXME: get magic number 8 from the current target
  max_partition_size = 8
  reduce_trip_cnt = nl.num_programs(list(program_reduce_axes))
  if reduce_trip_cnt > max_partition_size and grad_out.buffer is nl.psum:
    # Move the data from PSUM to SB
    grad_out = nl.copy(grad_out, dtype=compute_dtype, mask=mask)

  # Reduce between kernels
  if program_reduce_axes:
    _sum = nl.all_reduce(_sum, op=np.add, dtype=compute_dtype,
                         program_axes=program_reduce_axes,
                         mask=mask)

  _sum = _sum.expand_dims(tuple(range(_sum.ndim, grad_out.ndim)))
  _subtract = nl.subtract(grad_out, _sum, dtype=compute_dtype, mask=mask)
  return nl.multiply(_subtract, fwd_out, mask=mask)


def rmsnorm_kernel(data, weight, compute_dtype,
                   tile_par_reduce_indices, tile_free_reduce_indices,
                   program_reduce_axes, n, epsilon, mask):
  if tile_par_reduce_indices:
    # FIXME: Assert reduce the whole free dimension if tile_free_reduce_indices present
    return rmsnorm_matmul_reduce_kernel(data, weight, compute_dtype=compute_dtype,
                                        tile_free_reduce_indices=tile_free_reduce_indices,
                                        program_reduce_axes=program_reduce_axes,
                                        n=n, epsilon=epsilon, mask=mask)

  assert tile_free_reduce_indices
  # F only reduce using VecEng
  return rmsnorm_tensor_reduce_kernel(data, weight, compute_dtype=compute_dtype,
                                      tile_reduce_indices=tile_free_reduce_indices,
                                      program_reduce_axes=program_reduce_axes,
                                      n=n, epsilon=epsilon, mask=mask)


def rmsnorm_matmul_reduce_kernel(data, weight, compute_dtype,
                                 tile_free_reduce_indices, program_reduce_axes,
                                 n, epsilon, mask):
  if data.buffer is nl.psum:
    # Move the data from PSUM to SB first
    data = nl.copy(data, mask=mask)

  # if tile_free_reduce_indices:
  #   square = nl.ndarray([data.shape[0], 1], dtype=compute_dtype, buffer=nl.sbuf)
  #   square = square.as_tile()
  #   nisa.activation_reduce(op=np.square, data=data, reduce_op=np.add,
  #                          reduce_res=square)
  # else:
  square = nisa.activation(op=np.square, data=data,
                           dtype=compute_dtype, mask=mask)
  if tile_free_reduce_indices:
    square = nisa.tensor_reduce(np.add, data=square,
                                axis=tile_free_reduce_indices,
                                dtype=compute_dtype, mask=mask)

  par_shape, *free_shapes = data.shape

  if n_elts(free_shapes) == 1 and program_reduce_axes:
    # VecEng + TensorEng reduction
    if program_reduce_axes:
      square_reduce = nl.all_reduce(square, op=np.add, dtype=compute_dtype,
                                    program_axes=program_reduce_axes,
                                    mask=mask)
    with sync_program(program_reduce_axes):
      mm_reduce_weight = nisa.memset((par_dim(par_shape), par_shape),
                                     value=1, dtype=compute_dtype)
      masked_square_reduce = square_reduce[mask] if mask else square_reduce
      square_reduce = nisa.nc_matmul(mm_reduce_weight, masked_square_reduce,
                                     is_stationary_onezero=True)
  else:
    # TensorEng reduction only
    mm_reduce_weight = nisa.memset((par_dim(par_shape), par_shape),
                                   value=1, dtype=compute_dtype)
    masked_square = square[mask] if mask else square
    square_reduce = nisa.nc_matmul(mm_reduce_weight, masked_square,
                                   is_stationary_onezero=True)
    if program_reduce_axes:
      square_reduce = nl.all_reduce(square_reduce, op=np.add,
                                    dtype=compute_dtype,
                                    program_axes=program_reduce_axes,
                                    mask=mask)

  # Only compute in one kernel, not all of them
  with sync_program(program_reduce_axes):
    bias_memset = nisa.memset((par_dim(par_shape), 1),
                              value=epsilon,
                              dtype=compute_dtype, mask=mask)
    # FIXME: let nki infer the f32 dtype when buffer is set to psum
    rrms = nisa.activation(op=nl.sqrt,
                           data=square_reduce,
                           bias=bias_memset,
                           scale=1 / n,
                           dtype=np.float32,
                           buffer=nl.psum, mask=mask)

  # FIXME: get magic number 512 from the current target
  max_free_vec_size = 512

  if tile_free_reduce_indices:
    weight_expand_dims = tuple(i for i in range(1, data.ndim)
                                if i not in tile_free_reduce_indices)
    if weight_expand_dims:
      weight = weight.expand_dims(weight_expand_dims)

  # put the two multiplies together to allow tensorizer to fuse them to one STT
  scaled_data = nl.multiply(weight, data, dtype=compute_dtype,
                            max_free_vec_size=max_free_vec_size, mask=mask)


  if tile_free_reduce_indices:
    rrms = rrms.expand_dims(tile_free_reduce_indices)

  return nl.multiply(rrms, scaled_data, mask=mask,
                     max_free_vec_size=max_free_vec_size)

def rmsnorm_tensor_reduce_kernel(data, weight, compute_dtype,
                                 tile_reduce_indices, program_reduce_axes,
                                 n, epsilon, mask):
  if weight.buffer is nl.psum:
    # Move the data from PSUM to SB first
    weight = nl.copy(weight, mask=mask)

  scaled_data = nl.multiply(weight, data, dtype=compute_dtype)
  rrms_ = _reciprocal_rms_exp(data, compute_dtype, tile_reduce_indices,
                              program_reduce_axes, n, epsilon, mask=mask)
  return nl.multiply(scaled_data, rrms_, mask=mask)


def _reciprocal_rms_exp(src, compute_dtype, tile_reduce_indices,
                        program_reduce_axes, n, epsilon, mask):
  if tile_reduce_indices:
    data = nl.ndarray((par_dim(src.shape[0]), 1), dtype=compute_dtype)
    data = data.as_tile()
    nisa.activation_reduce(op=np.square, data=src, reduce_op=np.add,
                           reduce_res=data)
  else:
    data = nisa.activation(op=np.square, data=src,
                           dtype=compute_dtype, mask=mask)

  # Reduce between spmd kernels on the same device
  if program_reduce_axes:
    data = nl.all_reduce(data, op=np.add, dtype=compute_dtype,
                         program_axes=program_reduce_axes,
                         mask=mask)
  bias = nisa.memset(data.shape, epsilon, dtype=data.dtype, mask=mask)
  rrms_ = nisa.activation(op=nl.sqrt, data=data, bias=bias,
                          scale=1 / n, dtype=src.dtype, mask=mask)
  rrms_ = rrms_.expand_dims(tuple(range(rrms_.ndim, src.ndim)))

  return rrms_
