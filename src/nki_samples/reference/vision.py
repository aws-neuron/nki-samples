"""
Copyright (c) 2023, Amazon.com. All Rights Reserved

Vision kernels - Builtin high performance Vision NKI kernels

"""
import numpy as np

import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc import nki
from neuronxcc.nki.language import par_dim
import neuronxcc.nki.typing as nt


@nki.jit
def select_and_scatter_kernel(operand_tensor, source_tensor):
  """
  Implementation of a select-and-scatter kernel.

  It selects an element from each window of operand_tensor, and then scatters
  source_tensor to the indices of the selected positions to construct out_tensor
  with the same shape as the operand_tensor.

  This kernel assumes that
   - windows dimensions:  (3, 3)
   - windows strides:     (2, 2)
   - padding:             (1, 1)
   - init value:          0
   - select computation:  greater-than
   - scatter computation: add

  IO Tensor layouts:
   - operand_tensor: shape   (n, c, h, w)
   - source_tensor : shape   (n, c, src_h, src_w)
   - out_tensor    : shape   (n, c, h, w)
  
  IO tensor dtypes:
   - This kernel assumes all IO tensors have the same dtype
  """
  N, C, H, W = operand_tensor.shape  # batch, channel, height, width
  sw_h, sw_w = (3, 3)  # set window dimensions to 3
  stride_h, stride_w = (2, 2) # set window strides to 2
  padding = (1, 1) # set padding to 2
  src_n, src_c, src_h, src_w = source_tensor.shape
  padded_h = H + sum(padding)
  padded_w = W + sum(padding)
  assert N == src_n and C == src_c
  assert (padded_h - sw_h) // stride_h + 1 == src_h
  assert (padded_w - sw_w) // stride_w + 1 == src_w
  assert H == W and src_h == src_w

  # restriction imposed when we work on two batches together in partitions 0-64 and 64-128
  assert C == 64 and N % 2 == 0

  kernel_dtype = operand_tensor.dtype
  assert operand_tensor.dtype == source_tensor.dtype

  out_tensor = nl.ndarray((N, C, H, W), dtype=operand_tensor.dtype,
                          buffer=nl.shared_hbm)

  p = 128  # num of partitions to use
  for ib in nl.affine_range(N // 2):
    win_idx_mask_local = nl.ndarray((par_dim(p), src_h, src_w, sw_h, sw_w),
                                    dtype=nl.uint8, buffer=nl.sbuf)

    operand_local = nl.full((par_dim(p), padded_h, padded_w),
                            dtype=kernel_dtype, buffer=nl.sbuf,
                            fill_value=nl.finfo(kernel_dtype).min)

    for ib_1 in nl.affine_range(2):
      # pad and load the first and second batch to partitions 0-64, 64-128
      operand_local[ib_1 * 64:(ib_1 + 1) * 64, 1:H+1, 1:W+1] = nl.load(
        operand_tensor[2 * ib + ib_1, 0:64, 0:H, 0:W], dtype=kernel_dtype)

    for ik in nl.affine_range(src_h):
      # All the instructions in this loop will work on the same `window`
      window = nl.mgrid[0:p, 0:src_w, 0:sw_h, 0:sw_w]

      # create sliding windows for operand_local
      operand_local_sw: nt.tensor[p, src_w, sw_h, sw_w] = nl.copy(
        operand_local[window.p, 2 * ik + window.y, 2 * window.x + window.z])

      """
      Step 1
      Iterate all sliding windows, and find the maximum val of each window.
      """
      max_val: nt.tensor[p, src_w, 1, 1] = nl.max(
        operand_local_sw, axis=[2, 3], dtype=kernel_dtype, keepdims=True)

      """
      Step 2
      Create a mask with the shape of all the sliding windows
      - When the a position value equal to the maximum of the its sliding windows, the mask value = 1
      - Otherwise, the mask value = 0
      """

      _win_idx_mask: nt.tensor[p, src_w, sw_h, sw_w] = nl.equal(
        max_val, operand_local_sw, dtype=nl.int8)

      """
      Step 3
      The sliding window in the mask may have multiple positions with value 1.
      Need to find the first position with value 1 which we will later scatter to that position.

      1. Generate a linear sequence with same size of the window using iota and
         get the maximum value of the sequence.
      2. Based on each mask window, we select the maximum value when the
         position value is 0, and the corresponding value from the linear sequence when
         the position value is 1.
      3. Then we get minimum value of each window. 
      """
      index = (3 * window.y + window.z)
      linear_indices: nt.tensor[p, src_w, sw_h, sw_w] = nisa.iota(index, dtype=nl.uint32)

      selecting_win_indices: nt.tensor[p, src_w, sw_h, sw_w] = nl.where(
        _win_idx_mask, linear_indices, sw_h * sw_w, dtype=nl.uint32)
      selected_win_indices: nt.tensor[p, src_w, 1, 1] = nl.min(
        selecting_win_indices, axis=[2, 3], dtype=nl.uint32, keepdims=True)

      """
      Step 4
      With the minimum value, we can create another mask where value 1 indicates the postition we are going to scatter to.
      """
      win_idx_mask_local[0:p, ik, 0:src_w, 0:sw_h, 0:sw_w] = nl.equal(
        selected_win_indices, linear_indices, dtype=nl.uint8)

    source_local = nl.ndarray((par_dim(p), src_h, src_w), dtype=kernel_dtype, buffer=nl.sbuf)
    for ib_1 in nl.affine_range(2):
      source_local[ib_1 * 64:(ib_1 + 1) * 64, 0:src_h, 0:src_w] = nl.load(
        source_tensor[2 * ib + ib_1, 0:64, 0:src_h, 0:src_w],
        dtype=kernel_dtype)

    vals_nonlocal = nl.ndarray((sw_h, sw_w, par_dim(p), H, W), dtype=kernel_dtype, buffer=nl.private_hbm)
    for iq in nl.affine_range(sw_h):
      for ih in nl.affine_range(sw_w):
        """
        Step 5
        Scatter on the windows using the mask created in step 4
        """

        tile = nl.mgrid[0:p, 0:src_h, 0:src_w]
        vals_local = nl.zeros((par_dim(p), padded_h, padded_w), dtype=kernel_dtype, buffer=nl.sbuf)
        vals_local[tile.p, 2 * tile.x + iq, 2 * tile.y + ih] = nl.multiply(
          source_local, win_idx_mask_local[tile.p, tile.x, tile.y, iq, ih],
          dtype=kernel_dtype)

        nl.store(vals_nonlocal[iq, ih, 0:p, 0:H, 0:W], value=vals_local[0:p, 1:H+1,  1:W+1])

    out_local = nl.zeros((par_dim(p), H, W), dtype=kernel_dtype, buffer=nl.sbuf, name='out_local')

    for iq in nl.affine_range(sw_h):
      for ih in nl.affine_range(sw_w):
        """
        Step 6
        Reduce add for each window from Step 5 to get the final result
        """
        vals_local: nt.tensor[p, H, W] = nl.load(vals_nonlocal[iq, ih, 0:p, 0:H, 0:W])
        out_local += vals_local

    for ib_1 in nl.affine_range(2):
      nl.store(out_tensor[2 * ib + ib_1, 0:64, 0:H, 0:W],
               value=out_local[(ib_1 * 64):((ib_1 + 1) * 64), 0:H, 0:W])

  return out_tensor


@nki.jit
def resize_nearest_fixed_dma_kernel(data_tensor, out_shape):
  """
  Resize the input image to the given size using the nearest interpolation mode. This kernel is designed to be used when the scaling factor is not an integer. 

  Example:
   - Input height : 30, Input width : 20
   - Output height : 59, Output width : 38

  IO tensor layouts:
   - data_tensor: shape   (in_b, in_h, in_w, in_c)
   - out_tensor: shape   (out_b, out_h, out_w, out_c)
   - b : batch, c : channel, h : height, w : width
   - This kernel requires in_b == out_b as input batch and output batch must be identical
   - This kernel requires in_c == out_c as input channel and output channel must be identical
  
  """
  in_b, in_h, in_w, in_c = data_tensor.shape
  out_b, out_h, out_w, out_c = out_shape
  out_tensor = nl.ndarray(out_shape, dtype=data_tensor.dtype,
                          buffer=nl.shared_hbm)

  assert in_b == out_b, "Input batch and output batch must be identical"
  assert in_c == out_c, "Input channel and output channel must be identical"

  # Generate map
  h_scale, w_scale = 1.0 * in_h / out_h, 1.0 * in_w / out_w
  h_map = np.floor(np.fromfunction(lambda i, _: i * h_scale, (out_h, out_w), dtype=np.float32))
  w_map = np.floor(np.fromfunction(lambda _, j: j * w_scale, (out_h, out_w), dtype=np.float32))
  map = (h_map * in_w + w_map).astype(np.int32).flatten()

  in_seqlen, out_seqlen = in_h * in_w, out_h * out_w

  data_tile = data_tensor.reshape(shape=(in_b, in_seqlen, in_c))
  out_tile = out_tensor.reshape(shape=(out_b, out_seqlen, out_c))

  b_map, c_map = nl.mgrid[0:in_b, 0:out_c]

  for i in nl.static_range(len(map)):
    target_addr = data_tile[b_map, map[i], c_map]
    local_data = nl.load(target_addr)
    dst_addr_0 = out_tile[b_map, i, c_map]
    nl.store(dst_addr_0, value=local_data)

  return out_tensor
