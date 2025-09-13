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

  # Scaling factors for height and width
  h_scale: float = 1.0 * in_h / out_h
  w_scale: float = 1.0 * in_w / out_w

  # Create flattened views
  in_seqlen: int = in_h * in_w
  out_seqlen: int = out_h * out_w
  data_view_flattened = data_tensor.reshape(shape=(in_b, in_seqlen, in_c))
  output_view_flattened = out_tensor.reshape(shape=(out_b, out_seqlen, out_c))

  # Tile configuration
  H_W_TILE_SIZE = 128
  H_W_NUM_TILES = (out_seqlen + H_W_TILE_SIZE - 1) // H_W_TILE_SIZE

  for b in nl.affine_range(out_b):
    for h_w_tile in nl.affine_range(H_W_NUM_TILES):
      # Partition dimension for image pixel spatial positions, free dimension for channels
      i_p_hw = nl.arange(H_W_TILE_SIZE)[:, None]
      i_f_channel = nl.arange(out_c)[None, :]

      # Output position for current tile
      tile_start = h_w_tile * H_W_TILE_SIZE
      output_pos = tile_start + i_p_hw
      output_pos_iota = nisa.iota(output_pos, dtype=nl.int32)

      # Convert flattened position to 2D coordinates
      out_h_idx = nl.floor(output_pos_iota / out_w, dtype=nl.int32)
      out_w_idx = nl.mod(output_pos_iota, out_w, dtype=nl.int32)

      # Compute nearest neighbor source indices
      src_h_idx = nl.floor(nl.multiply(out_h_idx, h_scale, dtype=nl.float32), dtype=nl.int32)
      src_w_idx = nl.floor(nl.multiply(out_w_idx, w_scale, dtype=nl.float32), dtype=nl.int32)

      # Convert to flattened index
      load_indices = src_h_idx * in_w + src_w_idx

      # Boundary mask
      valid_mask = (output_pos < out_seqlen)

      # Gather and store
      target_addr = data_view_flattened[b, load_indices, i_f_channel]
      loaded_tile_sbuf = nl.load(target_addr, mask=valid_mask)

      nl.store(
          output_view_flattened[b, output_pos, i_f_channel],
          value=loaded_tile_sbuf,
          mask=valid_mask
      )

  return out_tensor


@nki.jit
def adaptive_avg_pool2d_kernel(in_tensor, output_size):
  """
  Implementation of adaptive average pooling 2D kernel.
  
  Applies a 2D adaptive average pooling over an input signal composed of several 
  input planes. The output size is determined by the output_size parameter, and 
  the kernel automatically computes the appropriate pooling regions for each output 
  element.

  This kernel implements the same functionality as PyTorch's AdaptiveAvgPool2d operation.
  
  IO Tensor layouts:
   - in_tensor: shape (N, C, H, W) - NCHW format
   - out_tensor: shape (N, C, OH, OW) - NCHW format
   - N: batch size, C: channels, H: height, W: width
   - OH, OW: output height and width determined by output_size
  
  Parameters:
   - in_tensor: Input tensor in NCHW format
   - output_size: Can be an integer or tuple of two integers
     - If integer: output will be square (output_size x output_size)
     - If tuple: output will be (OH x OW) where OH, OW = output_size
  
  IO tensor dtypes:
   - This kernel supports float32, float16, and bfloat16 dtypes
  """
  N, C, H, W = in_tensor.shape
  
  # Handle output_size parameter
  if isinstance(output_size, int):
    OH = OW = output_size
  else:
    OH, OW = output_size
  
  # Validate output size
  assert OH > 0 and OW > 0, "Output size must be positive"
  assert OH <= H and OW <= W, "Output size cannot be larger than input size"
  
  # Create output tensor
  out_tensor = nl.ndarray((N, C, OH, OW), dtype=in_tensor.dtype, buffer=nl.shared_hbm)
  
  # Compute the start and end indices for each pooling window
  # These arrays define the pooling regions for each output position
  h_start_indices = (np.arange(OH, dtype=np.int32) * H) // OH
  h_end_indices = ((np.arange(1, OH + 1, dtype=np.int32) * H + OH - 1) // OH)
  w_start_indices = (np.arange(OW, dtype=np.int32) * W) // OW
  w_end_indices = ((np.arange(1, OW + 1, dtype=np.int32) * W + OW - 1) // OW)
  h_spans = h_end_indices - h_start_indices
  w_spans = w_end_indices - w_start_indices
  
  # Tile configuration for processing N*C dimension
  NC_TILE_SIZE = min(N * C, nl.tile_size.pmax)
  n_nc_tiles = (N * C + NC_TILE_SIZE - 1) // NC_TILE_SIZE
  
  # Spatial tile sizes to fit within SBUF memory constraints
  # SBUF limit is 192KB per partition, using conservative tile sizes
  POOL_H_TILE_SIZE = 128
  POOL_W_TILE_SIZE = 128
  
  # Flatten batch and channel dimensions for efficient processing
  in_flattened = in_tensor.reshape(shape=(N * C, H, W))
  out_flattened = out_tensor.reshape(shape=(N * C, OH, OW))
  
  # Allocate buffers with spatial tiling support
  max_pool_h = min(int(h_spans.max()), POOL_H_TILE_SIZE)
  max_pool_w = min(int(w_spans.max()), POOL_W_TILE_SIZE)
  
  in_tile_sbuf = nl.zeros((n_nc_tiles, nl.par_dim(NC_TILE_SIZE), max_pool_h, max_pool_w), 
                          dtype=in_tensor.dtype, buffer=nl.sbuf)
  out_tile_sbuf = nl.zeros((n_nc_tiles, nl.par_dim(NC_TILE_SIZE), OH, OW), 
                           dtype=in_tensor.dtype, buffer=nl.sbuf)

  # Process tiles along the N*C dimension
  for nc_tile_idx in nl.affine_range(n_nc_tiles):
    nc_tile_start = nc_tile_idx * NC_TILE_SIZE
    
    # Create partition indices for current tile
    i_p_nc = nl.arange(NC_TILE_SIZE)[:, None, None] + nc_tile_start
    
    # Compute adaptive average for each output position
    for out_h_idx in nl.static_range(OH):
      for out_w_idx in nl.static_range(OW):
        # Define the pooling window for this output position
        h_start, h_end = h_start_indices[out_h_idx], h_end_indices[out_h_idx]
        w_start, w_end = w_start_indices[out_w_idx], w_end_indices[out_w_idx]
        pool_h_size = h_end - h_start
        pool_w_size = w_end - w_start
        
        accumulator = nl.zeros((NC_TILE_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf)
        total_elements = pool_h_size * pool_w_size
        
        n_pool_h_tiles = (pool_h_size + POOL_H_TILE_SIZE - 1) // POOL_H_TILE_SIZE
        n_pool_w_tiles = (pool_w_size + POOL_W_TILE_SIZE - 1) // POOL_W_TILE_SIZE
        
        # Process pooling region in tiles
        for pool_h_tile_idx in nl.sequential_range(n_pool_h_tiles):
          cur_pool_h_tile_size = min(pool_h_size, POOL_H_TILE_SIZE)
          pool_h_tile_start = pool_h_tile_idx * cur_pool_h_tile_size
          i_f_pool_h = nl.arange(cur_pool_h_tile_size)[None, :, None] + pool_h_tile_start + h_start
          
          for pool_w_tile_idx in nl.sequential_range(n_pool_w_tiles):
            cur_pool_w_tile_size = min(pool_w_size, POOL_W_TILE_SIZE)
            pool_w_tile_start = pool_w_tile_idx * cur_pool_w_tile_size
            i_f_pool_w = nl.arange(cur_pool_w_tile_size)[None, None, :] + pool_w_tile_start + w_start
            
            # Load current pooling region tile
            nisa.dma_copy(
              src=in_flattened[nc_tile_start : nc_tile_start + NC_TILE_SIZE,
                               h_start + pool_h_tile_start : h_start + pool_h_tile_start + cur_pool_h_tile_size,
                               w_start + pool_w_tile_start : w_start + pool_w_tile_start + cur_pool_w_tile_size],
              dst=in_tile_sbuf[nc_tile_idx, :, 0:cur_pool_h_tile_size, 0:cur_pool_w_tile_size],
              mask=((i_p_nc < N * C) & (i_f_pool_h < H) & (i_f_pool_w < W))
            )
            
            # Sum elements in current tile
            tile_sum = nl.sum(
              in_tile_sbuf[nc_tile_idx, :, 0:cur_pool_h_tile_size, 0:cur_pool_w_tile_size],
              axis=[-2, -1],
              mask=((i_f_pool_h < H) & (i_f_pool_w < W))
            )
            
            # Accumulate sum
            accumulator[:, :] = accumulator[:, :] + tile_sum
        
        # Compute average
        if total_elements > 0:
          out_tile_sbuf[nc_tile_idx, :, out_h_idx, out_w_idx] = nl.divide(accumulator, total_elements)
        else:
          out_tile_sbuf[nc_tile_idx, :, out_h_idx, out_w_idx] = 0.0
    
    # Store output tile
    nl.store(
      out_flattened[nc_tile_start:nc_tile_start + NC_TILE_SIZE, :, :],
      value=out_tile_sbuf[nc_tile_idx],
      mask=(i_p_nc < N * C)
    )
  
  return out_tensor
