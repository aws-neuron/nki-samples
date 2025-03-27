"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

Kernels written by the AWS Neuron.

WARNING: These kernels:
   - Are tested only against internal nightly builds
   - May not be compatible with public NeuronSDK releases
   - Have not been extensively tested across all input configurations
   - Carry no compatibility guarantees
   - The behavior of these kernels may be modified without prior notice

Convolution kernels

"""
import math
import neuronxcc.nki as nki
import numpy as np
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc.nki.language import par_dim
from common import div_ceil
from transpose import transpose_to_last_dim, tiled_dve_transpose_10, tiled_dve_transpose_210
from neuronxcc.nki.language.constants import sizeinbytes
from typing import Optional


def next_power_of_two(x):
  """
  If x is a power of two, return x.
  Else, return the next power of two.
  """
  return 1 << (x - 1).bit_length()

def reshape_all(refs, shapes):
  new_refs = []
  for ref, shape in zip(refs, shapes):
    new_refs.append(ref.reshape(shape) if ref.shape != shape else ref)
  return new_refs

def write_dilated_tensor(src, dest, dilation):
  i_H = nl.arange(src.shape[-2])[:, None]
  i_W = nl.arange(src.shape[-1])[None, :]
  tmp_src = nl.load(src[i_H, i_W])

  d_H = nl.arange(dest.shape[-2])[:, None]
  d_W = nl.arange(dest.shape[-1])[None, :]
  tmp_dest = nl.load(dest[d_H, d_W])
  for i_p in range(src.shape[0]):
    for i_f in range(src.shape[1]):
      tmp_dest[i_H + i_H * (dilation[0] - 1), i_W + i_W * (dilation[1] - 1)] += tmp_src[i_H, i_W]


def calc_dilated_dimension(d_size, dil):
  return (d_size - 1) * (dil - 1) + d_size

def calc_inverse_dilated_dimension(dilated_size, dil):
  return (dilated_size + dil - 1) // dil

def conv2d_depthwise_fb10_o01i_bf01(img_ref_fb10, filter_ref, out_ref, **kwargs): 
  img_ref = tiled_dve_transpose_210(img_ref_fb10)
  conv2d_depthwise_f01b_o01i_bf01(img_ref, filter_ref, out_ref, **kwargs)   

def conv2d_depthwise_f01b_o01i_bf01(img_ref, filter_ref, out_ref, **kwargs): 
  padding = kwargs['padding']
  H_padding_l, H_padding_r = padding[0]
  W_padding_l, W_padding_r = padding[1]
  assert W_padding_l >= 0 and W_padding_r >= 0 and H_padding_l >= 0 and H_padding_r >= 0, "Only supports positive paddings"
 
  assert kwargs['stride'] == [1, 1] or kwargs['stride'] == None or kwargs['stride'] == (1, 1), "Only support stride of size 1"
  assert kwargs['rhs_dilation'] == [1, 1] or kwargs['rhs_dilation'] == None or kwargs['rhs_dilation'] == (1, 1), "rhs_dilation is not supported when lhs_dilation is enabled"
 
  in_C, in_H, in_W, N = img_ref.shape #fb01_oi01→bf01
  k_FO, k_H, k_W, k_FI = filter_ref.shape
  _N, _C_out, out_H, out_W = out_ref.shape
  assert in_C == k_FO and _C_out == k_FO
  assert N == _N and 1 == k_FI, "depthwise conv only"

  lhs_dilation = kwargs['lhs_dilation']
  if lhs_dilation == None:
    lhs_dilation = (1, 1)

  if isinstance(lhs_dilation, int):
    lhs_dilation=(lhs_dilation, lhs_dilation)

  dil_H = calc_dilated_dimension(in_H, lhs_dilation[0]) + H_padding_l + H_padding_r
  dil_W = calc_dilated_dimension(in_W, lhs_dilation[1]) + W_padding_l + W_padding_r

  max_sbuf_size_in_bytes = 16 * 1024 

  h_tile_count = div_ceil(max_sbuf_size_in_bytes, dil_W * N * img_ref.dtype.itemsize) # find_highest_sbuf_partition_size_that_fits()
  max_h_tile_size = max(min(in_H, calc_inverse_dilated_dimension(h_tile_count, lhs_dilation[0]) - 1), k_H)

  max_channel_size = nl.tile_size.pmax
  num_channel_tiles, channel_tile_size = div_ceil(in_C, max_channel_size), min(in_C, max_channel_size)

  reduction_identity_matrix = nl.load(nl.shared_identity_matrix(min(in_C, channel_tile_size)))

  for channel_tile in nl.affine_range(num_channel_tiles):
    i_f_h = nl.arange(k_H)[None, :, None, None]
    i_f_w = nl.arange(k_W)[None, None, :, None]
    i_batch = nl.arange(N)[None, None, None, :]
    i_channel = nl.arange(channel_tile_size)[:, None, None, None]
    channel_mask = (i_channel + channel_tile * channel_tile_size < in_C) 

    filter_sbuf = nl.load(filter_ref[i_channel + channel_tile * channel_tile_size, i_f_h, i_f_w, 0], mask=channel_mask)
    broadcasted_filter = nl.ndarray(shape=(channel_tile_size, out_W, k_H, k_W, N), dtype=filter_sbuf.dtype)
    for w in nl.affine_range(out_W):
      i_c, i_o, i_h, i_w, i_n = nl.mgrid[0:channel_tile_size, 0:out_W, 0:k_H, 0:k_W, 0:N]
      broadcasted_filter[i_c, w, i_h, i_w, i_n] = filter_sbuf.reshape((channel_tile_size, 1, k_H, k_W, 1))

    # iterate over each row, loading stride window vertically and the entire row horizontally
    out_iter = iter(nl.static_range(out_H))
    for outer_row in out_iter:
      is_last_row = outer_row > out_H - H_padding_r

      h_tile_size = max_h_tile_size if not is_last_row else out_H - outer_row
      i_h = nl.arange(h_tile_size)[None, :, None, None]
      i_w = nl.arange(in_W)[None, None, :, None]

      org_image_index = max(0, calc_inverse_dilated_dimension(outer_row + 1 - H_padding_l, lhs_dilation[0]) - 1)
      # a dilation row is a row added to the original image during dilation, which means that it does not actually contain any data apart from 0s
      # so when loading original image, we need to advance original's index by 1, as that's where the next data row we are going to see.
      # imaging the following:
      # 
      # original image:
      # A
      # B
      # C
      #
      # dilated image:
      # A
      # 0 <---
      # B
      # 0 
      # C
      #
      # if we currently were on first 0 of the dilated image, A is the "parent" row of our current row, while B is the next row we would want to load
      # from the original image in order to construct its dilated version
      is_current_row_a_dilation_row = (outer_row + 1 + lhs_dilation[0] - 1 - H_padding_l) % lhs_dilation[0] != 0 and outer_row >= H_padding_l
      # how many rows we need to leave blank in order to simulate a dilated tile
      dilation_offset = 0
      if is_current_row_a_dilation_row:
        # 2 below is constructed from :
        # 1 for converting index into the number of elements contained inside current tile, as calc_dilated_dimension expects a count of elements, not index
        # 1 for getting the index of the next element
        #
        # subsctract 1 at the end to convert back to an index and add padding because we want to work with dilated image indexes that do account for padding
        next_non_dilated_row_index = calc_dilated_dimension(org_image_index + 2, lhs_dilation[0]) - 1 + H_padding_l
        current_dilated_row_index = outer_row 
        dilation_offset = next_non_dilated_row_index - current_dilated_row_index
        # index of the next data element we want to load
        org_image_index += 1
        # adjust size of current h tile to reflect one less element to load
        i_h = nl.arange(max(1, h_tile_size - 1))[None, :, None, None]
      # if we know that we are at the last element of the original image, there is nothing else to load.
      if org_image_index + h_tile_size >= in_H:
        i_h = nl.arange(h_tile_size - (org_image_index + h_tile_size - in_H))[None, :, None, None]

      image_sbuf = nl.load(
          img_ref[i_channel + channel_tile * channel_tile_size, i_h + org_image_index, i_w, i_batch],
          mask=(channel_mask & (i_h + org_image_index < in_H)),
      )

      dilated_h_size = calc_dilated_dimension(h_tile_size, lhs_dilation[0]) if not is_last_row else dil_H - outer_row
      i_padding = 0
      skip_copy = False
      if outer_row < H_padding_l:
        dilated_h_size += H_padding_l - outer_row
        i_padding = H_padding_l - outer_row
      if outer_row > out_H - H_padding_r:
        trailing_padding = outer_row + 1 - (out_H - H_padding_r)
        # TODO: if this is true, there isn't going to be any work done on remainding of the current tile from this point forward,
        # so might as well terminate the loop
        skip_copy = trailing_padding >= k_H
      i_padding += dilation_offset

      dilated_image = nl.zeros(
        (channel_tile_size, dilated_h_size, dil_W, N),
        dtype=img_ref.dtype,
        buffer=nl.sbuf,
        name="dilated_image",
      )

      if not skip_copy:
        dilated_image[i_channel, i_h + i_h * (lhs_dilation[0] - 1) + i_padding, i_w + i_w * (lhs_dilation[1] - 1) + W_padding_l, i_batch] = image_sbuf[i_channel, i_h, i_w, i_batch]

      window_size = (dilated_h_size - k_H) + 1
      window_size = window_size if outer_row + window_size < out_H else window_size - (outer_row + window_size - out_H)
      results_accumulation_sbuf = nl.ndarray((channel_tile_size, window_size, out_W, N), dtype=dilated_image.dtype, buffer=nl.sbuf)

      for h_offset in nl.affine_range(window_size):
        broadcasted_image = nl.ndarray(shape=(channel_tile_size, out_W, k_H, k_W, N), dtype=dilated_image.dtype)
        
        for w in nl.affine_range(out_W):
          i_b_c = nl.arange(channel_tile_size)[:, None, None, None]
          i_b_h = nl.arange(k_H)[None, :, None, None] 
          i_b_w = nl.arange(k_W)[None, None, :, None]
          i_b_n = nl.arange(N)[None, None, None, :]

          broadcasted_image[i_b_c, w, i_b_h, i_b_w, i_b_n] = dilated_image[i_b_c, i_b_h+h_offset, i_b_w+w, i_b_n]

        tensor_product = nisa.tensor_tensor(broadcasted_filter, broadcasted_image, np.multiply).reshape((channel_tile_size, out_W, k_H*k_W, N))

        if out_W * N > nl.tile_size.psum_fmax:
          if out_W > nl.tile_size.psum_fmax:
            num_n_tiles, n_tile_size = N, 1
          else:
            num_n_tiles = nl.tile_size.psum_fmax // out_W
            n_tile_size = div_ceil(N, num_n_tiles)
        else:
            num_n_tiles, n_tile_size = 1, N

        num_psum_tiles, psum_tile_size = div_ceil(out_W, nl.tile_size.psum_fmax), min(out_W, nl.tile_size.psum_fmax)

        for n_tile in nl.affine_range(num_n_tiles):
          for psum_tile in nl.affine_range(num_psum_tiles):
            # technically using ndarray here is incorrect, but tensorizer will recognize it as accumulation, so it does not matter
            # and we can skip zeros memset
            reduced_tensor = nl.ndarray(shape=(channel_tile_size, psum_tile_size, n_tile_size), dtype=np.float32, buffer=nl.psum)
            for kernel in nl.affine_range(k_W*k_H):
              i_channel = nl.arange(channel_tile_size)[:, None, None]
              i_w = nl.arange(psum_tile_size)[None, :, None]
              i_n = nl.arange(n_tile_size)[None, None, :]

              reduced_tensor[i_channel, i_w, i_n] += nisa.nc_matmul(
                reduction_identity_matrix,
                tensor_product[i_channel, i_w + psum_tile * psum_tile_size, kernel, i_n + n_tile * n_tile_size][(i_w + psum_tile * psum_tile_size < out_W) & (i_n + n_tile * n_tile_size < N)],
              )

            i_a_c = nl.arange(channel_tile_size)[:, None, None]
            i_a_w = nl.arange(psum_tile_size)[None, :, None] + psum_tile * psum_tile_size
            i_a_n = nl.arange(n_tile_size)[None, None, :] + n_tile * n_tile_size

            results_accumulation_sbuf[i_a_c, h_offset, i_a_w, i_a_n] = nisa.tensor_copy(
              reduced_tensor,
              mask=((i_a_w < out_W) & (i_a_n < N)),
            )

      i_h_window = nl.arange(window_size)[None, :, None, None] 
      i_o_w = nl.arange(out_W)[None, None, :, None]
      for batch in nl.affine_range(N):
        i_b = nl.arange(1)[None, None, None, :] + batch
        _ = nl.store(
          out_ref[batch, i_channel + channel_tile * channel_tile_size, i_h_window + outer_row, i_o_w],
          results_accumulation_sbuf[i_channel, i_h_window, i_o_w, i_b],
          mask=(channel_mask)
        )

      # FIXME
      for i in nl.static_range(window_size - 1):
        # don't advance iterator over the last iteration, since out loop is going to do it for us
        # skip this many rows, since we've already computerd them
        _ = next(out_iter, -1)

def conv2d_pbp_fb01_io01_01bf_kernel_experimental_1(img_ref, filter_ref, out_ref, padding, stride, rhs_dilation=None, groups=None, **kwargs):
  """
  kernel_img_input_desc='fb01', kernel_filter_desc='io01', kernel_output_desc='01bf'    
  """
  # Load b,01, pftranspose to 01,b, write back?
  IMG_F, IMG_B, IMG_0, IMG_1 = img_ref.shape

  t_img_hbm = nl.ndarray((IMG_0, IMG_F, IMG_1, IMG_B), dtype=img_ref.dtype, buffer=nl.hbm)

  # For now i'll just load b / 1 for simplicity

  IMG_B_NUM_TILES, IMG_B_TILE_SIZE = div_ceil(IMG_B, 128), min(IMG_B, 128)
  IMG_1_NUM_TILES, IMG_1_TILE_SIZE = div_ceil(IMG_1, 128), min(IMG_1, 128)

  #img_local_tile = nl.ndarray((par_dim(IMG_1_TILE_SIZE), IMG_B_TILE_SIZE), dtype=img_ref.dtype, buffer=nl.sbuf)

  for idx_f in nl.affine_range(IMG_F):
    for idx_0 in nl.affine_range(IMG_0):
      for idx_b in nl.affine_range(IMG_B_NUM_TILES):
        for idx_1 in nl.affine_range(IMG_1_NUM_TILES):
          ip_img = nl.arange(IMG_B_TILE_SIZE)[:, None]
          if_img = nl.arange(IMG_1_TILE_SIZE)[None, :]
          ip_img_t = nl.arange(IMG_1_TILE_SIZE)[:, None]
          if_img_t = nl.arange(IMG_B_TILE_SIZE)[None, :]
          
          img_local = nl.load(
                img_ref[idx_f, IMG_B_TILE_SIZE * idx_b + ip_img, idx_0, IMG_1_TILE_SIZE * idx_1 + if_img], 
                mask = (idx_b * IMG_B_TILE_SIZE + ip_img < IMG_B) & (idx_1 * IMG_1_TILE_SIZE + if_img < IMG_1),
              )
          
          img_local_tile = nisa.nc_transpose(img_local, mask=(idx_b * IMG_B_TILE_SIZE + ip_img < IMG_B) & (idx_1 * IMG_1_TILE_SIZE + if_img < IMG_1))

          nl.store(t_img_hbm[idx_0, idx_f, IMG_1_TILE_SIZE * idx_1 + ip_img_t, IMG_B_TILE_SIZE * idx_b + if_img_t], img_local_tile[ip_img_t, if_img_t], 
                   mask=(idx_b * IMG_B_TILE_SIZE + if_img_t < IMG_B) & (idx_1 * IMG_1_TILE_SIZE + ip_img_t < IMG_1)
          )



  # Load o, 01, pftranspose to 01,o, write back?
  # For now i'll just load o / 1 for simplicity.
  FILTER_I, FILTER_O, FILTER_0, FILTER_1 = filter_ref.shape

  FILTER_O_NUM_TILES, FILTER_O_TILE_SIZE = div_ceil(FILTER_O, 128), min(FILTER_O, 128)
  FILTER_1_NUM_TILES, FILTER_1_TILE_SIZE = div_ceil(FILTER_1, 128), min(FILTER_1, 128)

  t_filter_hbm = nl.ndarray((FILTER_0, FILTER_I, FILTER_1, FILTER_O), dtype=filter_ref.dtype, buffer=nl.hbm, name='t_filter_hbm')


  for idx_i in nl.affine_range(FILTER_I):
    for idx_0 in nl.affine_range(FILTER_0):
      for idx_o in nl.affine_range(FILTER_O_NUM_TILES):
        for idx_1 in nl.affine_range(FILTER_1_NUM_TILES):

          ip_filter = nl.arange(FILTER_O_TILE_SIZE)[:, None]
          if_filter = nl.arange(FILTER_1_TILE_SIZE)[None, :]
          ip_filter_t = nl.arange(FILTER_1_TILE_SIZE)[:, None]
          if_filter_t = nl.arange(FILTER_O_TILE_SIZE)[None, :]

          filter_local = nl.load(
                filter_ref[idx_i, FILTER_O_TILE_SIZE * idx_o + ip_filter, idx_0, FILTER_1_TILE_SIZE * idx_1 + if_filter],
                mask = (idx_o * FILTER_O_TILE_SIZE + ip_filter < FILTER_O) & (idx_1 * FILTER_1_TILE_SIZE + if_filter < FILTER_1),
              )
          
          filter_local_tile = nisa.nc_transpose(filter_local,  mask=(idx_o * FILTER_O_TILE_SIZE + ip_filter < FILTER_O) & (idx_1 * FILTER_1_TILE_SIZE + if_filter < FILTER_1))

          nl.store(t_filter_hbm[idx_0, idx_i, FILTER_1_TILE_SIZE * idx_1 + ip_filter_t, FILTER_O_TILE_SIZE * idx_o + if_filter_t], filter_local_tile[ip_filter_t, if_filter_t],
                   mask=(idx_o * FILTER_O_TILE_SIZE + if_filter_t < FILTER_O) & (idx_1 * FILTER_1_TILE_SIZE + ip_filter_t < FILTER_1)
          )



  # For output, it is 01fb. Load fb, pftranspose to bf, write back?
  OUT_0, OUT_1, OUT_B, OUT_F = out_ref.shape

  OUT_F_NUM_TILES, OUT_F_TILE_SIZE = div_ceil(OUT_F, 128), min(OUT_F, 128)
  OUT_B_NUM_TILES, OUT_B_TILE_SIZE = div_ceil(OUT_B, 128), min(OUT_B, 128)

  t_out_hbm = nl.ndarray((OUT_0, OUT_1, OUT_F, OUT_B), dtype=out_ref.dtype, buffer=nl.hbm)

  conv2d_pbp_0f1b_0i1o_01fb_experimental_1(t_img_hbm, t_filter_hbm, t_out_hbm, padding, stride)

  for idx_0 in nl.affine_range(OUT_0):
    for idx_1 in nl.affine_range(OUT_1):
      for idx_f in nl.affine_range(OUT_F_NUM_TILES):
        for idx_b in nl.affine_range(OUT_B_NUM_TILES):
          ip_out = nl.arange(OUT_F_TILE_SIZE)[:, None]
          if_out = nl.arange(OUT_B_TILE_SIZE)[None, :]
          ip_out_t = nl.arange(OUT_B_TILE_SIZE)[:, None]
          if_out_t = nl.arange(OUT_F_TILE_SIZE)[None, :]

          out_local = nl.load(
                t_out_hbm[idx_0, idx_1, OUT_F_TILE_SIZE * idx_f + ip_out, OUT_B_TILE_SIZE * idx_b + if_out],
                mask = (idx_f * OUT_F_TILE_SIZE + ip_out < OUT_F) & (idx_b * OUT_B_TILE_SIZE + if_out < OUT_B),
          )

          out_local_tile = nisa.nc_transpose(out_local,  mask=(idx_f * OUT_F_TILE_SIZE + ip_out < OUT_F) & (idx_b * OUT_B_TILE_SIZE + if_out < OUT_B))
          nl.store(
            out_ref[idx_0, idx_1, OUT_B_TILE_SIZE * idx_b + ip_out_t, OUT_F_TILE_SIZE * idx_f + if_out_t],
            out_local_tile,
            mask=(idx_f * OUT_F_TILE_SIZE + if_out_t < OUT_F) & (idx_b * OUT_B_TILE_SIZE + ip_out_t < OUT_B)
          )


def conv2d_pbp_0f1b_0i1o_01fb_experimental_1(img_ref, filter_ref, out_ref, padding, stride, rhs_dilation=None, groups=None, **kwargs):
  """
  Experimental kernel for convolutions.

  Prefetches filter

  kernel_img_input_desc='0f1b', kernel_filter_desc='0i1o', kernel_output_desc='01fb'
  """
  print(f"Running conv2d_pbp_0f1b_0i1o_01fb_experimental_1, img_ref.shape: {img_ref.shape}, filter_ref.shape: {filter_ref.shape}, out_ref.shape: {out_ref.shape}")
  kernel_dtype = img_ref.dtype

  if isinstance(padding[0], (list, tuple)):
    padding = [p[0] for p in padding]

  # Shapes
  H_img, C_in, W_img, N_batch = img_ref.shape
  H_filter, c_in_assert, W_filter, C_out = filter_ref.shape
  assert c_in_assert == C_in, "c_in_assert != C_in"
  H_out, W_out, c_out_assert, batch_assert = out_ref.shape
  assert c_out_assert == C_out, "c_out_assert != C_out"
  assert batch_assert == N_batch, "batch_assert != N_batch"
  
  # Stride
  H_stride, W_stride = stride
  assert tuple(stride) == (1, 1), "stride != (1, 1)"

  # Padding
  H_padding, W_padding = padding
  assert (H_img + 2 * H_padding - H_filter) // H_stride + 1 == H_out, f"H_out: {H_out}, (H_img + 2 * H_padding - H_filter) // H_stride + 1: {(H_img + 2 * H_padding - H_filter) // H_stride + 1}"
  assert (W_img + 2 * W_padding - W_filter) // W_stride + 1 == W_out, f"W_out: {W_out}, (W_img + 2 * W_padding - W_filter) // W_stride + 1: {(W_img + 2 * W_padding - W_filter) // W_stride + 1}"

  # Tiling
  # TODO: How am i gonna handle padding here?


  C_out_num_tiles, C_out_tile_size = div_ceil(C_out, 128), min(C_out, 128)  # Corresponds to matmul RHS free
  N_batch_num_tiles, N_batch_tile_size = div_ceil(N_batch, 512), min(N_batch, 512)  # Corresponds to matmul LHS free

  # matmul contraction dimension
  # For simplicity, we just use H_filter
  assert H_filter <= 128, "H_filter > 128" # TODO: To handle this case, tile H_filter
  # Note: If H_filter is small, this is suboptimal. We may want to put C_in in P as well, or even
  # possibly put W_filter in P. But for now, we won't allow that (we will treat these as batch)

  # Note for reader: The name is a tad confusing. Despite being called `W_filter_prefetch`,
  # this means "W_filter coordinate" for purposes of "prefetching" _image_ (not filter!!!)
  W_filter_prefetch_inner = min(W_filter, 8) # FIXME: Add logic to choose best value (possibly even 1 if N_batch is big)
  W_filter_prefetch_outer = div_ceil(W_filter, W_filter_prefetch_inner)
 
  print(f"W_filter: {W_filter} W_filter_prefetch_inner: {W_filter_prefetch_inner} W_filter_prefetch_outer: {W_filter_prefetch_outer}")


  # Used to speed up "memset 0"
  zero_slice = nl.zeros((par_dim(H_filter), max(C_out_tile_size, N_batch_tile_size)), dtype=kernel_dtype, name='zero_slice')

  for c_out_tile in nl.affine_range(C_out_num_tiles):

    # FIXME: If we modified the filter transpose logic to take a different input shape, could optimize further?
    # FIXME: In cases where `H_filter, C_in` are too big so that prefetch of full filter won't work,
    # consider treating part of the contract dim(s) as batch for whole kernel (i.e. move to be outermost loop)
    filter_local_prefetch = nl.ndarray((par_dim(H_filter), W_filter, C_in, C_out_tile_size), dtype=kernel_dtype, name='filter_local_prefetch')
    
    for w_filter in nl.affine_range(W_filter):
      for c_in in nl.affine_range(C_in):
        ip_contract = nl.arange(H_filter)[:, None]
        if_filter = nl.arange(C_out_tile_size)[None, :] # Becomes ip_out (matmult LHS free -> result P dim)

        filter_local_prefetch[ip_contract, w_filter, c_in, if_filter] = nl.load(
              filter_ref[ip_contract, c_in, w_filter, C_out_tile_size * c_out_tile + if_filter], # HBM shape: (H_filter, C_in, W_filter, C_out)
              mask = (ip_contract < H_filter) & (C_out_tile_size * c_out_tile + if_filter < C_out)
            )

    for h_out in nl.affine_range(H_out):
      for w_out in nl.affine_range(W_out):
        # Compute output[h_out, w_out, :, :]

        # To compute output[h_out, w_out, :, :], we need the entire filter and a window into the input
        # The "size" of the window into the input is the filter size.

        # out_sbuf = nl.zeros((C_out_num_tiles, N_batch_num_tiles, par_dim(C_out_tile_size), N_batch_tile_size), dtype=kernel_dtype)

        out_sbuf = nl.zeros((par_dim(C_out_tile_size), N_batch_num_tiles, N_batch_tile_size), dtype=np.float32, buffer=nl.sbuf, name='out_sbuf')

        for c_in in nl.affine_range(C_in):
          for w_filter_pref_outer in nl.affine_range(W_filter_prefetch_outer):
            prefetch_load_ip = nl.arange(H_filter)[:, None]
            prefetch_load_if = nl.arange(W_filter_prefetch_inner * N_batch)[None, :]

            h_img = h_out*H_stride + ip_contract - H_padding
            # c_in is as above
            wn_img = w_out*W_stride*N_batch - W_padding*N_batch + w_filter_pref_outer*W_filter_prefetch_inner*N_batch + prefetch_load_if
            
            img_local_pref = nl.ndarray((par_dim(H_filter), W_filter_prefetch_inner * N_batch), dtype=kernel_dtype, name='img_local_pref_hacknoinit')
            
            img_ref_view = img_ref.reshape((H_img, C_in, W_img * N_batch))
            img_local_pref[prefetch_load_ip, prefetch_load_if] = nl.load(
              img_ref_view[h_img, c_in, wn_img],
              mask = (h_img >= 0) & (h_img < H_img) & (wn_img >= 0) & (wn_img < W_img * N_batch)
            )

            for n_batch_tile in nl.affine_range(N_batch_num_tiles):
              out_psum = nl.zeros((par_dim(C_out_tile_size), N_batch_tile_size), dtype=np.float32, buffer=nl.psum, name='out_psum')

              for w_filter_pref_inner in nl.affine_range(W_filter_prefetch_inner):
                ip_contract = nl.arange(H_filter)[:, None]  # Partition index into contract dimension which is the h coordinate of the filter

                if_img = nl.arange(N_batch_tile_size)[None, :]  #
                                                                # Becomes if_out (matmult RHS free (corresp to tensorizer "lhs") -> result F dim)
                
                if_filter = nl.arange(C_out_tile_size)[None, :] # Becomes ip_out (matmult LHS free (corresp to tensorizer "rhs") -> result P dim)
                ip_out = nl.arange(C_out_tile_size)[:, None]

                # Used for masking, affine-only (no `arange` style APs) expression that yields `w_filter` value (w coordinate into filter tensor) 
                w_filter_both_affine = w_filter_pref_outer * W_filter_prefetch_inner + w_filter_pref_inner

                lhs_mask = (
                    # Ensure w_filter isn't reading too much (`W_filter_pref_inner` may be too big for last iteration of w_filter_pref_outer loop)
                    (w_filter_both_affine < W_filter)
                    # Ensure if_filter isn't reading too much (`C_out_tile_size` may be too big for last iteration of c_out_tile loop)
                    & (c_out_tile * C_out_tile_size + if_filter < C_out)

                    # Contract restrictions
                    & (h_out*H_stride + ip_contract - H_padding >= 0)
                    & (h_out*H_stride + ip_contract - H_padding < H_img)
                  )
                
                rhs_mask = (
                    # Ensure h and w are within image bounds
                    # (h_img >= 0)
                    (h_out*H_stride + ip_contract - H_padding >= 0)
                    # & (h_img < H_img)
                    & (h_out*H_stride + ip_contract - H_padding < H_img)
                    # & (wn_img >= 0)
                    & (w_out*W_stride*N_batch - W_padding*N_batch + w_filter_pref_outer*W_filter_prefetch_inner*N_batch + w_filter_pref_inner*N_batch + n_batch_tile*N_batch_tile_size + if_img >= 0)
                    # & (wn_img < W_img * N_batch)
                    & (w_out*W_stride*N_batch - W_padding*N_batch + w_filter_pref_outer*W_filter_prefetch_inner*N_batch + w_filter_pref_inner*N_batch + n_batch_tile*N_batch_tile_size + if_img <  W_img * N_batch)

                    # Ensure w_filter isn't reading too much (`W_filter_pref_inner` may be too big for last iteration of w_filter_pref_outer loop)
                    & (w_filter_both_affine < W_filter)

                    # Ensure if_img isn't reading too much (`N_batch_tile_size` might be too big for last iteration of n_batch_tile loop)
                    & (n_batch_tile*N_batch_tile_size + if_img < N_batch)

                    # Ensure if_img isn't reading too much (`N_batch_tile_size` might be too big for last iteration of _both_ n_batch_tile and w_filter_pref_outer loops)
                    & (w_filter_pref_inner*N_batch + n_batch_tile*N_batch_tile_size + if_img <  W_filter_prefetch_inner * N_batch)
                  )

                out_psum += nisa.nc_matmul(
                  filter_local_prefetch[ip_contract, w_filter_both_affine, c_in, if_filter][lhs_mask], # Tensorizer "RHS"
                  img_local_pref[ip_contract, w_filter_pref_inner*N_batch + n_batch_tile*N_batch_tile_size + if_img][rhs_mask], # Tensorizer "LHS"
                )

              if_out = nl.arange(N_batch_tile_size)[None, :]
              out_sbuf[ip_out, n_batch_tile, if_out] += out_psum[ip_out, if_out]

        # FIXME: Can merge into single dma?
        for n_batch_tile in nl.affine_range(N_batch_num_tiles):
          ip_out = nl.arange(C_out_tile_size)[:, None]
          if_out = nl.arange(N_batch_tile_size)[None, :]
          nl.store(
            out_ref[h_out, w_out, C_out_tile_size * c_out_tile + ip_out, N_batch_tile_size * n_batch_tile + if_out],
            out_sbuf[ip_out, n_batch_tile, if_out],
            mask = (C_out_tile_size * c_out_tile + ip_out < C_out) & (N_batch_tile_size * n_batch_tile + if_out < N_batch)
          )

@nki.jit
def conv2d(
    in_ref, 
    filter_ref, 
    out_ref, 
    padding: list[list[int]] = [[0,0],[0,0]], 
    srcs_shapes: Optional[list[int]] = None, 
    dsts_shapes: Optional[list[int]] = None, 
    stride: list[int] = [1, 1], 
    # lhs_dilation: Optional[tuple[int]] = None, 
    rhs_dilation: tuple[int] = (1, 1),
    in_perm: list[int] = [0, 1, 2, 3],
    kern_perm: list[int] = [0, 1, 2, 3],
    out_perm: list[int] = [0, 1, 2, 3]
    ):
  """NKI kernel to compute a 2D convolution, where filter_ref convolves over in_ref to produce out_ref.

  Args:
      in_ref: an input tensor, of dimensions N (batch), C (channels), H (height), W (width) -- the order of these dimensions depends on in_perm
      filter_ref: a filter/kernel tensor, of dimensions i (input channels), o (output channels), f_H (filter height), f_W (filter width) -- the order of these dimensions depends on kern_perm
      out_ref: an output tensor of dimensions N, out_C, out_H, out_W -- the order of these dimensions depends on out_perm
      padding: [[H_padding_l, H_padding_r], [W_padding_l, W_padding_r]] specifies amount of padding on in_ref 
      srcs_shapes: (optional) shapes of in_ref and filter_ref
      dsts_shapes: (optional) shapes of out_ref
      stride:[h_stride, w_stride] specifies stride of filter/kernel tensor over in_ref
      rhs_dilation: (h_dilation, w_dilation) specifies dilation of in_ref
      in_perm: specifies permutation of N, C, H, W which defines the layout of in_ref
      kern_perm: specifies permutation of i, o, f_H, f_W which defines the layout of filter_ref
      out_perm: specifies permutation of N, out_C, out_H, out_W which defines the layout of out_ref

  Return:
      out_ref: the resulting convolution output tensor
  """
  kwargs = {
    'padding': padding, 
    'srcs_shapes': srcs_shapes,
    'dsts_shapes': dsts_shapes,
    'stride': stride,
    # 'lhs_dilation': lhs_dilation,
    'rhs_dilation': rhs_dilation,
    'in_perm': in_perm,
    'kern_perm': kern_perm,
    'out_perm': out_perm
  }
  conv2d_dw_fb01_io01_01bf_rep_nhwc_Pcinh(in_ref, filter_ref, out_ref, **kwargs)


"""
Naming convention:
dw: derivative weight
fb01_io01_01bf: input and output shapes
rep: using replication strategy
nhwc: how nchw is being transposed into for the matmul
Pcinh: cin and h are partition axes

(In the backward) This strategy is useful when c, k0, k1 is small and w, h is large. For example:
[7,7,3,64] = convolution([16,3,224,224], [16,64,112,112]), 
window={size=112x112 pad=3_2x3_2 rhs_dilate=2x2}, dim_labels=fb01_io01->01bf

Replication can be replaced later with the real replication mode DMA.

Pesudocode:
for h_f // h_rep:
  for c_in, w_f, c_out:
    # replication and dilation
    filter_local[c_in, h, cout] = load weight[c_in, h_f, w_f, cout]
  for n, k0, c_in, w_f, k1: 
    # reloading on k0; coalesse on the last dim (w_f/k1)
    # replication and padding
    img_local[c_in, h_f, k0, w_f, n] = load ifmap[c_in, h_f+k0, w_f, n]
  for c_out, n, k1, c_in, k0:
      out[k0, k1, n, cout] += 
          img_local[c_in, h_f, k0, k1, c_in] * filter_local[c_in, c_out, h]
"""

def conv2d_dw_fb01_io01_01bf_rep_nhwc_Pcinh(img_ref, filter_ref, out_ref, **kwargs):
  padding = kwargs['padding']
  H_padding_l, H_padding_r = padding[0]
  W_padding_l, W_padding_r = padding[1]
  srcs_shapes = kwargs.get('srcs_shapes', None)
  dsts_shapes = kwargs.get('dsts_shapes', None)
  stride = kwargs.get('stride', [1, 1])
  h_stride, w_stride = stride
  lhs_dilation = kwargs.get('lhs_dilation', None) # unsupported
  rhs_dilation = kwargs.get('rhs_dilation', None)
  in_perm = kwargs.get('in_perm', None)
  kern_perm = kwargs.get('kern_perm', None)
  out_perm = kwargs.get('out_perm', None)

  nchw_in = in_perm == [0, 1, 2, 3]
  nchw_out = out_perm == [0, 1, 2, 3]

  kernel_dtype = img_ref.dtype

  if rhs_dilation is None:
    rhs_dilation = (1, 1)

  if srcs_shapes:
    img_ref, filter_ref = reshape_all([img_ref, filter_ref], srcs_shapes)
  if dsts_shapes:
    out_ref, = reshape_all([out_ref], dsts_shapes)

  # transpose to 0, 3, 1, 2 - C_in, H_f, W_f, C_out
  if kern_perm == [0, 3, 1, 2]:
    C_out, H_f, W_f, C_in = filter_ref.shape
    _weight = nl.ndarray(shape=(C_in, H_f, W_f, C_out), dtype=kernel_dtype, buffer=nl.hbm, name='weight_transposed')
    transpose_to_last_dim(filter_ref.reshape((C_out, (H_f * W_f), C_in)), dim=0, dst=_weight)
  elif kern_perm == [0, 1, 2, 3]:
    C_out, C_in, H_f, W_f = filter_ref.shape
    _weight = nl.ndarray(shape=(C_in, H_f, W_f, C_out), dtype=kernel_dtype, buffer=nl.hbm, name='weight_transposed')
    transpose_to_last_dim(filter_ref.reshape((C_out, C_in*H_f*W_f)), dim=0, dst=_weight)
  elif kern_perm == [1, 0, 3, 2]:
    C_in, C_out, W_f, H_f = filter_ref.shape
    _weight = nl.ndarray(shape=(C_in, H_f, W_f, C_out), dtype=kernel_dtype, buffer=nl.hbm, name='weight_transposed')
    # FIXME: is this the most performant way to do 2 transposes?
    _intermediate_weight = nl.ndarray(shape=(C_in, C_out, H_f, W_f), dtype=kernel_dtype, buffer=nl.hbm, name='_intermediate_weight_transposed')
    transpose_to_last_dim(filter_ref.reshape((C_in, C_out, W_f, H_f)), dim=2, dst=_intermediate_weight) 
    transpose_to_last_dim(_intermediate_weight.reshape((C_in, C_out, H_f, W_f)), dim=1, dst=_weight)  
  else:
    assert kern_perm == [1, 0, 2, 3]
    C_in, C_out, H_f, W_f = filter_ref.shape
    _weight = nl.ndarray(shape=(C_in, H_f, W_f, C_out), dtype=kernel_dtype, buffer=nl.hbm, name='weight_transposed')
    transpose_to_last_dim(filter_ref.reshape((C_in, C_out, H_f, W_f)), dim=1, dst=_weight)

  # transpose to 3, 0, 1, 2 - C_in, H, W, N
  if in_perm == [0, 3, 1, 2]:
    assert all([s == 1 for s in stride]), "unsupported perm with strides"
    N, H, W, C_in = img_ref.shape
    _ifmap = nl.ndarray(shape=(C_in, H, W, N), dtype=kernel_dtype, buffer=nl.hbm, name='ifmap_transposed')
    transpose_to_last_dim(img_ref.reshape((N, H*W, C_in)), dim=0, dst=_ifmap) 
  elif nchw_in:
    assert all([s == 1 for s in stride]), "unsupported perm with strides"
    N, C_in, H, W = img_ref.shape
    _ifmap = img_ref
  elif in_perm == [3, 0, 1, 2]:
    C_in, H, W, N = img_ref.shape
    _ifmap = img_ref
  else:
    C_in, N, H, W = img_ref.shape
    _ifmap = nl.ndarray(shape=(C_in, H, W, N), dtype=kernel_dtype, buffer=nl.hbm, name='ifmap_transposed')
    transpose_to_last_dim(img_ref.reshape((C_in, N, H, W)), dim=1, dst=_ifmap) 

  # create conv_out in 1, 2, 3, 0 - C_out, K0, K1, N
  if nchw_out:
    _, _, K0, K1 = out_ref.shape
  elif out_perm == [1, 2, 3, 0]:
    _, K0, K1, _ = out_ref.shape
    conv_out = out_ref
  else:
    K0, K1, _, _ = out_ref.shape
    conv_out = nl.ndarray(shape=(C_out, K0, K1, N), dtype=kernel_dtype, buffer=nl.hbm, name='conv_out')

  canonical_H_f, canonical_W_f = canonicalize_filter_shape(H_f, W_f, rhs_dilation)

  # need to add tiling to remove the belowed restriction
  # avoid predicates in the inner tile of replication if we have dilation in rhs
  #  instead, allow tiling on the outer tile of replication
  H_REP = replication_factor(canonical_H_f, C_in, rhs_dilation[0])
  # either no replication or divisible
  assert H_REP == 1 or H_REP % rhs_dilation[0] == 0

  if H_REP == 1 and rhs_dilation[0] > 1:
    # we need to tile for rhs_dilation[0] on H_INNER
    H_OUTER_NUM_TILES, H_OUTER_TILE_SIZES, _ = tile(canonical_H_f, rhs_dilation[0])
  else:
    H_OUTER_NUM_TILES, H_OUTER_TILE_SIZES, _ = tile(canonical_H_f, H_REP)

  # computation tiles
  COUT_NUM_TILES, COUT_TILE_SIZES, _ = tile(C_out, 128)

  # tiling for lhs
  tile_size = 512

  if nchw_in:
    # N cannot be chosen as LHS free, so only tile on K0 and K1
    N_COMP_NUM_TILES, N_COMP_TILE_SIZES = 1, 1 # computation tile
    N_DMA_NUM_TILES, N_DMA_TILE_SIZES = 1, 1 # only prefetch on H and W
    N_OUTER_NUM_TILES, N_OUTER_TILE_SIZES = N, 1
  else:
    # we will only need to tile one of these
    N_COMP_NUM_TILES, N_COMP_TILE_SIZES, tile_size = tile(N, tile_size) # computation tile
    N_DMA_NUM_TILES, N_DMA_TILE_SIZES = 1, N # prefetch on N and W
    N_OUTER_NUM_TILES, N_OUTER_TILE_SIZES = 1, N

  K1_NUM_TILES, K1_TILE_SIZES, tile_size = tile(K1, tile_size)
  K0_NUM_TILES, K0_TILE_SIZES, _ = tile_with_stride(K0*h_stride, tile_size, h_stride)
  if K0_TILE_SIZES == 1: # stride happens inter tile
    K0_COMP_NUM_TILES, K0_COMP_TILE_SIZES = K0, 1
    h_stride_intra_tile = h_stride
  else:                  # stride happens intra tile
    K0_COMP_NUM_TILES, K0_COMP_TILE_SIZES = K0_NUM_TILES, div_ceil(K0_TILE_SIZES, h_stride)
    h_stride_intra_tile = 1

  # prefetching tiles
  # only tile on prefetching W_f for simplicity
  PREFETCH_TILE_SIZE = 512*16 # TODO: pick a better tile size here
  WF_NUM_TILES, WF_TILE_SIZES = (W_f, 1) if C_out > PREFETCH_TILE_SIZE / 2 else \
    (div_ceil(W_f, PREFETCH_TILE_SIZE // C_out), min(W_f, PREFETCH_TILE_SIZE // C_out))
  print(f'W_f: {W_f}: {WF_NUM_TILES} * {WF_TILE_SIZES}')

  # for debugging only: we can determine the lhs from above:
  lhs_frees = list(map(lambda p: f'{p[0]} - {p[1]}', 
                       filter(lambda p: p[1] > 1, 
                              zip([('K0', K0), ('K1', K1), ('N', N)], [K0_TILE_SIZES, K1_TILE_SIZES, N_COMP_TILE_SIZES]))))
  print(f'config: {C_in}, {N}, {H}, {W}; LHS: {(K0, K1, N)}, LHS_FREES: {lhs_frees}')
  name = f'{C_in}, {N}, {H}, {W}; {lhs_frees}'.replace('"', '').replace('\'', '')

  for n_outer_tile in nl.affine_range(N_OUTER_NUM_TILES):
    out_sb = nl.zeros((COUT_NUM_TILES, K0_COMP_NUM_TILES, K1_NUM_TILES, N_COMP_NUM_TILES, par_dim(COUT_TILE_SIZES), K0_TILE_SIZES, K1_TILE_SIZES, N_COMP_TILE_SIZES), 
                        dtype=kernel_dtype, buffer=nl.sbuf, name=f'a0_sb_{name}')
  
    for h_outer in nl.affine_range(H_OUTER_NUM_TILES):
      # if there is negative padding on W: load a larger image, then make a copy to trim the padding 
      W_l_pos_padding = max(0,W_padding_l)
      W_r_pos_padding = max(0,W_padding_r)
      img_local_prefetch_raw = nl.zeros(shape=(N_DMA_NUM_TILES, K0_NUM_TILES, nl.par_dim(C_in*H_REP), K0_TILE_SIZES, W+W_l_pos_padding+W_r_pos_padding, N_DMA_TILE_SIZES), dtype=kernel_dtype, buffer=nl.sbuf, name='a0_img_local_prefetch')
      img_local_prefetch = nl.zeros(shape=(N_COMP_NUM_TILES, K0_NUM_TILES, nl.par_dim(C_in*H_REP), K0_TILE_SIZES, W+W_padding_l+W_padding_r, N_DMA_TILE_SIZES), dtype=kernel_dtype, buffer=nl.sbuf, name='a0_img_local_prefetch_neg')
      for n_tile in nl.affine_range(N_DMA_NUM_TILES):
        for k0_tile in nl.affine_range(K0_NUM_TILES):
          for h_rep in nl.affine_range(H_REP):
            # we cannot handle NEGATIVE padding on i_w because it will result in predicate i_w >= -w_padding_l or i_w < W+W_padding_r and bubble in free dim
            # so we need to have a tensor copy to make this legal
            i_cin, i_k0, i_w, i_n = create_indices(C_in, K0_TILE_SIZES, W, N_DMA_TILE_SIZES)

            h = h_outer * H_OUTER_TILE_SIZES + h_rep
            k0 = k0_tile * K0_TILE_SIZES * h_stride_intra_tile + i_k0
            n = n_outer_tile * N_OUTER_TILE_SIZES + n_tile * N_DMA_TILE_SIZES + i_n

            # replication on h, implicit padding on H, explicit padding on W, prefetchig on W
            # all W padding is non-negative, no boundary check needed: 
            #   i_w+W_l_pos_padding>=0 and i_w+W_l_pos_padding<W+W_l_pos_padding+W_r_pos_padding
            mask = (h+k0-H_padding_l < H) & (h+k0-H_padding_l >= 0) & (n < N)

            if nchw_in:
              img_local_prefetch_raw[n_tile, k0_tile, i_cin + h_rep * C_in, i_k0, i_w+W_l_pos_padding, i_n] = nl.load(_ifmap[n, i_cin, h+k0-H_padding_l, i_w], mask=mask) 
            else:
              img_local_prefetch_raw[n_tile, k0_tile, i_cin + h_rep * C_in, i_k0, i_w+W_l_pos_padding, i_n] = nl.load(_ifmap[i_cin, h+k0-H_padding_l, i_w, n], mask=mask)

        if is_negative_padding(padding[1]):
          for k0_tile in nl.affine_range(K0_NUM_TILES):
            i_1, i_k0, i_w, i_n = create_indices(C_in*H_REP, K0_TILE_SIZES, W+W_padding_l+W_padding_r, N_DMA_TILE_SIZES)

            # offset i_w on src if W_padding_l<0
            W_padding_l_offset = max(0,-W_padding_l)
            img_local_prefetch[n_tile, k0_tile, i_1, i_k0, i_w, i_n] = img_local_prefetch_raw[n_tile, k0_tile, i_1, i_k0, i_w+W_padding_l_offset, i_n]
        else:
          img_local_prefetch = img_local_prefetch_raw

      # The filter is usually bigger in training, and we will need to tile on W_f. Consider the filter's free axes are H_f, W_f and C_out. 
      # And the free axes for image is H, W, N. We have H_f ~= H, W_f ~= W, and C_out >> N. (N is always small)
      # When C_out is too big, we may overflow SB when prefetching the whole W_f. We need to tile it, and better to fuse it the matmul.
      # In inference, H_f and W_f is small, but since N is also small so we likely can afford prefetching on the whole W on the image
      # Therefore, we prefetch in the image above, and then fuse filter dma with matmul. 
      for wf_tile in nl.affine_range(WF_NUM_TILES):
        filter_local_prefetch = nl.zeros((par_dim(C_in*H_REP), WF_TILE_SIZES, C_out), dtype=kernel_dtype, name='a0_filter_local_prefetch')

        if H_REP == 1:  # rhs_dilation happen on H_OUTER
          H_REP_NUM_TILES = 1
        else:  # rhs_dilation happen on H_REP
          assert H_REP // rhs_dilation[0]
          H_REP_NUM_TILES = H_REP // rhs_dilation[0] # should be always divisible

        for h_rep in nl.affine_range(H_REP_NUM_TILES): 
          i_cin = nl.arange(C_in)[:, None, None]
          i_w_f = nl.arange(WF_TILE_SIZES)[None, :, None] # prefetching on W_f and C_out
          i_cout = nl.arange(C_out)[None, None, :]

          h = h_outer * H_REP_NUM_TILES + h_rep
          wf = wf_tile * WF_TILE_SIZES + i_w_f
          mask = (h < H_f) & (wf < W_f)

          # this following dilates on W because (1) filter_local_prefetch is
          # memset to zero (2) h_rep represent a tile of [C_in, C_in*0, C_in*0]
          # where *0 from rhs_dilation.
          filter_local_prefetch[i_cin + C_in * h_rep * rhs_dilation[0], i_w_f, i_cout] = nl.load(_weight[i_cin, h, wf, i_cout], mask=mask)

        for k0_tile in nl.affine_range(K0_COMP_NUM_TILES):
          for k1_tile in nl.affine_range(K1_NUM_TILES):
            for n_tile in nl.affine_range(N_COMP_NUM_TILES):
              for c_out_tile in nl.affine_range(COUT_NUM_TILES):
                ps = nl.zeros(shape=(par_dim(COUT_TILE_SIZES), K0_COMP_TILE_SIZES, K1_TILE_SIZES, N_COMP_TILE_SIZES), dtype=np.float32, buffer=nl.psum, name=f'a0_psum_{name}')
                for w in nl.affine_range(WF_TILE_SIZES):
                  i_cin, i_k0, i_k1, i_n = create_indices(C_in*H_REP, K0_COMP_TILE_SIZES, K1_TILE_SIZES, N_COMP_TILE_SIZES)

                  k1 = k1_tile * K1_TILE_SIZES + i_k1
                  n = n_tile * N_COMP_TILE_SIZES + i_n
                  wf = wf_tile * WF_TILE_SIZES + w
                  if nchw_in:
                    img_local = img_local_prefetch[n_tile, k0_tile, i_cin, i_k0 * h_stride, wf*rhs_dilation[1]+k1 * w_stride, i_n] # strided by w_stride
                  else:
                    img_local = img_local_prefetch[0, k0_tile, i_cin, i_k0 * h_stride, wf*rhs_dilation[1]+k1 * w_stride, n] # strided by w_stride

                  _i_cin = nl.arange(C_in*H_REP)[:, None] # replicated
                  i_cout = nl.arange(COUT_TILE_SIZES)[None, :]
                  c_out = c_out_tile*COUT_TILE_SIZES+i_cout
                  filter_local = filter_local_prefetch[_i_cin, w, c_out]

                  i_cout_out = nl.arange(COUT_TILE_SIZES)[:, None, None]
                  ps[i_cout_out, i_k0, i_k1, i_n] += nisa.nc_matmul(
                    filter_local[c_out < C_out],
                    img_local[k1 < K1][n < N][wf < W_f],
                  )

                i_cout_out, i_k0, i_k1, i_n = create_indices(COUT_TILE_SIZES, K0_COMP_TILE_SIZES, K1_TILE_SIZES, N_COMP_TILE_SIZES)
                out_sb[c_out_tile, k0_tile, k1_tile, n_tile, i_cout_out, i_k0, i_k1, i_n] += ps[i_cout_out, i_k0, i_k1, i_n]

    # storing the compute results
    for k0_tile in nl.affine_range(K0_COMP_NUM_TILES):
      for k1_tile in nl.affine_range(K1_NUM_TILES):
        for n_tile in nl.affine_range(N_COMP_NUM_TILES):
          for c_out_tile in nl.affine_range(COUT_NUM_TILES):
            i_cout, i_k0, i_k1, i_n = create_indices(COUT_TILE_SIZES, K0_COMP_TILE_SIZES, K1_TILE_SIZES, N_COMP_TILE_SIZES)

            c_out = c_out_tile * COUT_TILE_SIZES + i_cout
            k0 = k0_tile * K0_COMP_TILE_SIZES + i_k0
            k1 = k1_tile * K1_TILE_SIZES + i_k1
            n = n_outer_tile * N_OUTER_TILE_SIZES + n_tile * N_COMP_TILE_SIZES + i_n
            mask = (c_out < C_out) & (k0 < K0) & (k1 < K1) & (n < N)

            nl.store(
              out_ref[n, c_out, k0, k1] if nchw_out else conv_out[c_out, k0, k1, n],
              out_sb[c_out_tile, k0_tile, k1_tile, n_tile, i_cout, i_k0, i_k1, i_n],
              mask=mask
            )

  if nchw_out:
    return
  if out_perm == [2, 3, 0, 1]:
    transpose_to_last_dim(conv_out, dim=0, dst=out_ref)
  else:
    assert out_perm == [1, 2, 3, 0]
    # no need to transpose


# FIXME: support nl.arange(1) in general
def create_indices(*tripcounts):
  rank = len(tripcounts)
  # rank needs to be reduced by 1 if last dim is 1
  # Note: may need to find all 1s toward the end
  if tripcounts[-1] == 1:
    assert tripcounts[-2] != 1, "Unhandled case"
    rank-=1
  indices = map(lambda c: nl.arange(c) if c > 1 else 0, tripcounts)

  indices = []
  colon = slice(None, None, None)
  cur_rank = 0
  for c in tripcounts:
    if c > 1:
      access = [None] * rank
      access[cur_rank] = colon
      indices.append(nl.arange(c)[tuple(access)])
    else:
      indices.append(0)
    cur_rank += 1
  return indices


# get the shape after applying dilation on the filter, 
def canonicalize_filter_shape(H_f, W_f, rhs_dilation=None):
  if rhs_dilation:
    H_f = (H_f - 1) * rhs_dilation[0] + 1
    W_f = (W_f - 1) * rhs_dilation[1] + 1
  return H_f, W_f

def is_negative_padding(padding):
  return any([p < 0 for p in padding])


# get the replication factor and try to avoid predicates on the 
#  inner tile by making sure the replication_factor % dilation == 0
def replication_factor(rep_from, rep_to, dilation=1):
  max_rep = 0
  for i in range(rep_from, 0, -1):
    if i * rep_to <= 128 and i % dilation == 0:
      max_rep = i
      break
  return max(max_rep, 1)


# return: of tiles, tile_size size, remaining tile size
def tile(tripcount, tile_size):
  if not tile_size:
    return tripcount, 1, 0
  return div_ceil(tripcount, tile_size), min(tripcount, tile_size), tile_size // tripcount


# adjust tilesize so that stride will not across two tiles
def tile_with_stride(tripcount, size, stride):
  if size < stride:
    # not enlarge size if:
    # (1) only the first elem is accessed in each tile
    # (2) TODO: if stride is too big - loading too many unused data
    return tripcount, 1, 0
  if size % stride != 0:
  # adjust tilesize so that stride will not across two tiles
    size = div_ceil(size, stride) * stride
  n_tiles, tile_size, remaining = tile(tripcount, size)
  assert tile_size % stride == 0, "wrong tilesize for striding"
  return n_tiles, tile_size, remaining

def conv1d_depthwise_default(img_ref, filter_ref, out_ref, **kwargs):
  padding = kwargs['padding']
  W_padding_l, W_padding_r = padding[1]

  N, C_in, H, W = img_ref.shape #bf01_oi01→bf01
  C_out, _, H_f, W_f = filter_ref.shape
  _N, _C_out, K0, K1 = out_ref.shape

  image_size = H * (W + W_padding_l + W_padding_r)
  window_size = H_f * W_f
  out_image_size = K0 * K1
  dtype = img_ref.dtype

  C_NUM_TILES, C_TILE_SIZE = div_ceil(C_in, 128), min(C_in, 128)
  
  img_local_prefetch_raw = nl.zeros(shape=(N, C_NUM_TILES, nl.par_dim(C_TILE_SIZE), image_size), dtype=dtype, buffer=nl.sbuf, name='a0_img_local_prefetch')
  for i_n in nl.affine_range(N):
    for c_tile in nl.affine_range(C_NUM_TILES):
      i_cin_tile, i_w = create_indices(C_TILE_SIZE, W)
      i_cin = i_cin_tile + c_tile * 128
      i_h = 0
      i_image = W_padding_l + i_w
      img_local_prefetch_raw[i_n, c_tile, i_cin_tile, i_image] = nl.load(img_ref[i_n, i_cin, i_h, i_w])

  filter_local = nl.zeros(shape=(C_NUM_TILES, nl.par_dim(C_TILE_SIZE), window_size), dtype=dtype, buffer=nl.sbuf, name='a0_filter_local')
  for c_tile in nl.affine_range(C_NUM_TILES):
    i_cin_tile, i_w = create_indices(C_TILE_SIZE, W_f)
    i_cin = i_cin_tile + c_tile * 128
    i_h = 0
    filter_local[c_tile, i_cin_tile, i_w * H_f + i_h] = nl.load(filter_ref[i_cin, i_h, i_h, i_w])

  out_sb = nl.zeros((N, C_NUM_TILES, par_dim(C_TILE_SIZE), out_image_size), 
                      dtype=dtype, buffer=nl.sbuf, name=f'output')
  
  for i_n in nl.affine_range(N):
    for c_tile in nl.affine_range(C_NUM_TILES):
      for i_out in nl.affine_range(out_image_size):
        i_p_a = nl.arange(C_TILE_SIZE)[:, None]
        i_f_a = nl.arange(W_f)[None, :]
        prod = nisa.tensor_tensor(img_local_prefetch_raw[i_n, c_tile, i_p_a, i_f_a+i_out], filter_local[c_tile, i_p_a, i_f_a], np.multiply) 
        out_sb[i_n, c_tile, i_p_a, i_out] = nisa.tensor_reduce(np.add, prod[i_p_a, i_f_a], axis=[1])

  for n in nl.affine_range(N):
    for c_tile in nl.affine_range(C_NUM_TILES):
      i_cout, i_k0, i_k1 = create_indices(C_TILE_SIZE, K0, K1)

      c_out = c_tile * C_TILE_SIZE + i_cout
      i_out = i_k1 * K0 + i_k0
      mask = (c_out < C_out)

      nl.store(
        out_ref[n, c_out, i_k0, i_k1],
        out_sb[n, c_tile, i_cout, i_out],
        mask=mask
      )
  return

def conv1d_depthwise_f_packing(img_ref, filter_ref, out_ref, **kwargs):
  padding = kwargs['padding']
  W_padding_l, W_padding_r = padding[1]

  N, C_in, H, W = img_ref.shape #bf01_oi01→bf01
  C_out, _, H_f, W_f = filter_ref.shape
  _N, _C_out, K0, K1 = out_ref.shape
  img_ref = img_ref.reshape([N,C_in,H*W])
  filter_ref = filter_ref.reshape([C_out, H_f*W_f])

  image_size = H * (W + W_padding_l + W_padding_r)
  window_size = H_f * W_f
  out_image_size = K0 * K1
  dtype = img_ref.dtype

  C_NUM_TILES, C_TILE_SIZE = div_ceil(C_in, 128), min(C_in, 128)
  
  # window_size has to be the first f for reduce, 
  # for optimization, we can pack extra from C_NUM_TILES, N and W. 
  # by observation, W is usually the largest out of the three
  flattend_input_img_f = div_ceil(512, window_size)
  flattend_input_img_size = div_ceil(K1, flattend_input_img_f)
 
  img_local_prefetch = nl.zeros(shape=(N, C_NUM_TILES, flattend_input_img_size, nl.par_dim(C_TILE_SIZE), window_size, flattend_input_img_f), dtype=dtype, buffer=nl.sbuf, name='a0_img_local')
  for i_n in nl.affine_range(N):
    for c_tile in nl.affine_range(C_NUM_TILES):
      img_local_prefetch_raw = nl.zeros((nl.par_dim(C_TILE_SIZE), image_size), dtype=dtype, buffer=nl.sbuf, name='a0_img_local_prefetch')
      i_cin_tile, i_w = create_indices(C_TILE_SIZE, W)
      i_cin = i_cin_tile + c_tile * 128
      i_h = 0
      i_image = W_padding_l + i_w
      img_local_prefetch_raw[i_cin_tile, i_image] = nl.load(img_ref[i_n, i_cin, i_w])
 
      for i_input_img_size in nl.affine_range(flattend_input_img_size):
        i_cin_tile, i_w, i_input_img_f = create_indices(C_TILE_SIZE, window_size, flattend_input_img_f)
        i_cin = i_cin_tile + c_tile * 128
        i_h = 0
        i_image = i_input_img_size * flattend_input_img_f  + i_w + i_input_img_f
        mask = (i_input_img_size * flattend_input_img_f + i_input_img_f < out_image_size)
        img_local_prefetch[i_n, c_tile, i_input_img_size, i_cin_tile, i_w, i_input_img_f] = nl.copy(img_local_prefetch_raw[i_cin_tile, i_image], mask=mask)

  filter_local = nl.zeros(shape=(C_NUM_TILES, nl.par_dim(C_TILE_SIZE), window_size), dtype=dtype, buffer=nl.sbuf, name='a0_filter_local')
  for c_tile in nl.affine_range(C_NUM_TILES):
    i_cin_tile, i_w = create_indices(C_TILE_SIZE, W_f)
    i_cin = i_cin_tile + c_tile * 128
    i_h = 0
    filter_local[c_tile, i_cin_tile, i_w * H_f + i_h] = nl.load(filter_ref[i_cin, i_w])
 
  for i_n in nl.affine_range(N):
    for c_tile in nl.affine_range(C_NUM_TILES):
      for i_input_img_size in nl.affine_range(flattend_input_img_size):
        i_cin_tile, i_w, i_input_img_f = create_indices(C_TILE_SIZE, window_size, flattend_input_img_f)
        i_image = i_input_img_size * flattend_input_img_f  + i_w + i_input_img_f
        mask = (i_input_img_size * flattend_input_img_f + i_input_img_f < out_image_size)
        prod = nisa.tensor_tensor(img_local_prefetch[i_n, c_tile, i_input_img_size, i_cin_tile, i_w, i_input_img_f], filter_local[c_tile, i_cin_tile, i_w], np.multiply, mask=mask) 
        out_sb = nisa.tensor_reduce(np.add, prod[i_cin_tile, i_w, i_input_img_f], axis=(1,), mask=mask)

        i_cin_tile, i_input_img_f = create_indices(C_TILE_SIZE, flattend_input_img_f)
        i_h = 0
        i_image = i_input_img_size * flattend_input_img_f  + i_input_img_f
        c_out = c_tile * C_TILE_SIZE + i_cin_tile
        mask = (c_out < C_out) & (i_image < out_image_size)
        nl.store(
          out_ref[i_n, c_out, i_h, i_image],
          out_sb[i_cin_tile, i_input_img_f],
          mask=mask
        )

  return

# bf01_oi01→bf01
# use a combination of tensor_tensor multiply and reduce to perform depthwise conv
# supports conv1d currently
def conv1d_depthwise_bf01_oi01_bf01(img_ref, filter_ref, out_ref, **kwargs):
  padding = kwargs['padding']
  H_padding_l, H_padding_r = padding[0]
  assert H_padding_l == 0 and H_padding_r == 0, "1d conv expected"
  W_padding_l, W_padding_r = padding[1]
  assert W_padding_l >= 0 and W_padding_r >= 0, "Only supports positive paddings"
 
  N, C_in, H, W = img_ref.shape #bf01_oi01→bf01
  C_out, _, H_f, W_f = filter_ref.shape
  _N, _C_out, K0, K1 = out_ref.shape
  assert C_in == C_out
  assert H == 1 and H_f == 1 and K0 == 1, "conv1d only"
 
  window_size = H_f * W_f
  assert window_size < 512, "expect small conv window"

  # window_size has to be the first f for reduce, 
  # for optimization, we can pack extra from C_NUM_TILES, N and W. 
  # by observation, W is usually the largest out of the three
  flattend_input_img_f = div_ceil(512, window_size)
  flattend_input_img_size = div_ceil(K1, flattend_input_img_f)
  if flattend_input_img_size == 1:
    return conv1d_depthwise_default(img_ref, filter_ref, out_ref, **kwargs)
  
  return conv1d_depthwise_f_packing(img_ref, filter_ref, out_ref, **kwargs)


"""
A conv kernel using the "column packing" strategy.

When we have conv(matmul), we can ALWAYS pack block axes or contraction axes on
PE by column. Note that orginally, we only use rhs free axes there.

If we pack block axes, e.g. lhs=2, rhs=3, contract=3, batch=3, the matmul will
be like: (ABC are used for each batches)
  INPUT         WEIGH
  |------|      |---------|
  |CCBBAA|      |AAABBBCCC|
  |CCBBAA|  =>  |AAABBBCCC|
  |CCBBAA|      |AAABBBCCC|
  |------|      |---------|
                  v
                OUTPUT
                |---------|
                |      CCC|
                |      CCC|
                |   BBB   |
                |   BBB   | . => mask and reduce to 3x6
                |AAA      |
                |AAA      |
                |---------|
Note that the blank elements in the resulted tensor are garbage data. 

Similarly, if we pack contract axes, the resulted tensor will be like:
(note that the packing factor is 3, and ABC are for different contract axes)
  INPUT         WEIGH
  |------|      |---------|
  |CCBBAA|      |AAABBBCCC|
  |CCBBAA|  =>  |AAABBBCCC|
  |CCBBAA|      |AAABBBCCC|
  |------|      |---------|
                  v
                OUTPUT
                |---------|
                |      CCC|
                |      CCC|
                |   BBB   |
                |   BBB   | => mask and reduce to 3x2
                |AAA      |
                |AAA      |
                |---------|

Either way, we reorganize the output with DVE by masking out the garbage data
and reduction and get the finally result.

Packing block axes strategy is particular useful on backward depthwise
convolution. Because it is batched and has rhs=1. And compared with MM packing,
we can even pack more. So for now, the following kernel only applies the first
strategy on a max of 128 batch window.
"""
def conv2d_column_packing(img_ref, filter_ref, out_ref, **kwargs):
  dtype = img_ref.dtype

  C_in, N, H, W = img_ref.shape
  C_in, C_out, H_f, W_f = filter_ref.shape
  K0, K1, _, _ = out_ref.shape

  batch_group_count = kwargs['batch_group_count']
  assert batch_group_count == N == C_out # TODO: relax me

  kwargs['out_perm'] = [0, 1, 2, 3]

  MAX_F = 18200  # FIXME: Use nki.language.constants once max free size is added
  MAX_H = (MAX_F // sizeinbytes(img_ref.dtype)) // W
  H_NUM_TILES = H // MAX_H
  REMAINDER_H_TILE = H % MAX_H

  BGC_TILES, BGC_TILE_SIZE = div_ceil(batch_group_count, 128), min(batch_group_count, 128)
  kwargs['dsts_shapes'] = [[BGC_TILE_SIZE, BGC_TILE_SIZE, K0, K1]]
  for bgc_tile in nl.affine_range(BGC_TILES): # assumption on batch_group_count == N == C_out
    sparse_out_ref = nl.ndarray(shape=(BGC_TILE_SIZE, BGC_TILE_SIZE, K0, K1), dtype=dtype, buffer=nl.hbm, name='sparse_out')
    sub_img = img_ref
    sub_filter = filter_ref
    if BGC_TILES > 1:
      sub_img = nl.ndarray(shape=(C_in, BGC_TILE_SIZE, H, W), dtype=dtype, buffer=nl.hbm, name='sub_img')
      sub_filter = nl.ndarray(shape=(C_in, BGC_TILE_SIZE, H_f, W_f), dtype=dtype, buffer=nl.hbm, name='sub_filter')
      for c_in in nl.affine_range(C_in):
        for h_tile in nl.affine_range(H_NUM_TILES):
          sub_img_sbuf = nl.load(img_ref[c_in, bgc_tile*BGC_TILE_SIZE:(bgc_tile+1)*BGC_TILE_SIZE, h_tile*MAX_H:(h_tile+1)*MAX_H, :])
          nl.store(sub_img[c_in, :, h_tile*MAX_H:(h_tile+1)*MAX_H, :], sub_img_sbuf)

        if REMAINDER_H_TILE > 0:
          sub_img_sbuf_remainder = nl.load(img_ref[c_in, bgc_tile*BGC_TILE_SIZE:(bgc_tile+1)*BGC_TILE_SIZE, H_NUM_TILES*MAX_H:H_NUM_TILES*MAX_H+REMAINDER_H_TILE, :])
          nl.store(sub_img[c_in, :, H_NUM_TILES*MAX_H:H_NUM_TILES*MAX_H+REMAINDER_H_TILE, :], sub_img_sbuf_remainder)

        sub_filter_sbuf = nl.load(filter_ref[c_in, bgc_tile*BGC_TILE_SIZE:(bgc_tile+1)*BGC_TILE_SIZE, :, :])
        nl.store(sub_filter[c_in, :, :, :], sub_filter_sbuf)
      kwargs['srcs_shapes'] = [[C_in, BGC_TILE_SIZE, H, W], [C_in, BGC_TILE_SIZE, H_f, W_f]]

    conv2d_dw_fb01_io01_01bf_rep_nhwc_Pcinh(sub_img, sub_filter, sparse_out_ref, **kwargs)

    # mask out garbage and reduce
    mask = nl.shared_constant(np.fromfunction(lambda i, j, k, l: (i == j).astype(np.uint8), (sparse_out_ref.shape), dtype=np.uint8))

    l = nl.load(sparse_out_ref[:, :, :, :])
    r = nl.load(mask[:, :, :, :])
    masked = l * r
    reduced = nl.ndarray(shape=(BGC_TILE_SIZE, K0*K1), dtype=dtype, buffer=nl.sbuf, name='reduced')
    i_bgc, i_k0, i_k1 = nl.mgrid[0:BGC_TILE_SIZE, 0:K0, 0:K1]
    reduced[i_bgc, i_k0 * K1 + i_k1] = nl.sum(masked, axis=1)
    transposed = nl.transpose(reduced)
    transposed = transposed.reshape([K0*K1, 1, BGC_TILE_SIZE])

    out_ref = out_ref.reshape([K0*K1, 1, C_out]) # assuming N // batch_group_count == 1
    nl.store(out_ref[:, :, bgc_tile*BGC_TILE_SIZE:(bgc_tile+1)*BGC_TILE_SIZE], transposed)

def conv2d_column_packing_io10(img_ref, filter_ref, out_ref, **kwargs):
  dtype = img_ref.dtype

  if kwargs['in_perm'] == [1, 0, 3, 2]:
    img_ref = tiled_dve_transpose_10(img_ref, permutation=[0, 1, 3, 2])

  C_in, N, H, W = img_ref.shape
  C_in, C_out, W_f, H_f = filter_ref.shape
  K0, K1, _, _ = out_ref.shape
 
  batch_group_count = kwargs['batch_group_count']
  assert batch_group_count == N == C_out # TODO: relax me
 
  kwargs['out_perm'] = [0, 1, 2, 3]
 
  MAX_F = 18200  # FIXME: Use nki.language.constants once max free size is added
  MAX_H = (MAX_F // sizeinbytes(img_ref.dtype)) // W
  H_NUM_TILES = H // MAX_H
  REMAINDER_H_TILE = H % MAX_H
 
  BGC_TILES, BGC_TILE_SIZE = div_ceil(batch_group_count, 128), min(batch_group_count, 128)
  kwargs['dsts_shapes'] = [[BGC_TILE_SIZE, BGC_TILE_SIZE, K0, K1]]
  for bgc_tile in nl.affine_range(BGC_TILES): # assumption on batch_group_count == N == C_out
    sparse_out_ref = nl.ndarray(shape=(BGC_TILE_SIZE, BGC_TILE_SIZE, K0, K1), dtype=dtype, buffer=nl.hbm, name='sparse_out')
    sub_img = img_ref
    sub_filter = filter_ref
    if BGC_TILES > 1:
      sub_img = nl.ndarray(shape=(C_in, BGC_TILE_SIZE, H, W), dtype=dtype, buffer=nl.hbm, name='sub_img')
      sub_filter = nl.ndarray(shape=(C_in, BGC_TILE_SIZE, W_f, H_f), dtype=dtype, buffer=nl.hbm, name='sub_filter')
      for c_in in nl.affine_range(C_in):
        for h_tile in nl.affine_range(H_NUM_TILES):
          sub_img_sbuf = nl.load(img_ref[c_in, bgc_tile*BGC_TILE_SIZE:(bgc_tile+1)*BGC_TILE_SIZE, h_tile*MAX_H:(h_tile+1)*MAX_H, :])
          nl.store(sub_img[c_in, :, h_tile*MAX_H:(h_tile+1)*MAX_H, :], sub_img_sbuf)
 
        if REMAINDER_H_TILE > 0:
          sub_img_sbuf_remainder = nl.load(img_ref[c_in, bgc_tile*BGC_TILE_SIZE:(bgc_tile+1)*BGC_TILE_SIZE, H_NUM_TILES*MAX_H:H_NUM_TILES*MAX_H+REMAINDER_H_TILE, :])
          nl.store(sub_img[c_in, :, H_NUM_TILES*MAX_H:H_NUM_TILES*MAX_H+REMAINDER_H_TILE, :], sub_img_sbuf_remainder)
 
        sub_filter_sbuf = nl.load(filter_ref[c_in, bgc_tile*BGC_TILE_SIZE:(bgc_tile+1)*BGC_TILE_SIZE, :, :])
        nl.store(sub_filter[c_in, :, :, :], sub_filter_sbuf)
      kwargs['srcs_shapes'] = [[C_in, BGC_TILE_SIZE, H, W], [C_in, BGC_TILE_SIZE, W_f, H_f]]
 
    sub_shape = sub_filter.shape
    conv2d_dw_fb01_io01_01bf_rep_nhwc_Pcinh(sub_img, sub_filter, sparse_out_ref, **kwargs)
 
    # mask out garbage and reduce
    mask = nl.shared_constant(np.fromfunction(lambda i, j, k, l: (i == j).astype(np.uint8), (sparse_out_ref.shape), dtype=np.uint8))
 
    l = nl.load(sparse_out_ref[:, :, :, :])
    r = nl.load(mask[:, :, :, :])
    masked = l * r
    reduced = nl.ndarray(shape=(BGC_TILE_SIZE, K0*K1), dtype=dtype, buffer=nl.sbuf, name='reduced')
    i_bgc, i_k0, i_k1 = nl.mgrid[0:BGC_TILE_SIZE, 0:K0, 0:K1]
    reduced[i_bgc, i_k0 * K1 + i_k1] = nl.sum(masked, axis=1)
    transposed = nl.transpose(reduced)
    transposed = transposed.reshape([K0*K1, 1, BGC_TILE_SIZE])
 
    out_ref = out_ref.reshape([K0*K1, 1, C_out]) # assuming N // batch_group_count == 1
    nl.store(out_ref[:, :, bgc_tile*BGC_TILE_SIZE:(bgc_tile+1)*BGC_TILE_SIZE], transposed)

"""
A conv kernel using the "column packing" strategy to implement dw convolution with batch 1.

"""
def conv2d_column_packing_1(img_ref, filter_ref, out_ref, **kwargs):
  dtype = img_ref.dtype

  N, C_in, H, W = img_ref.shape
  C_out, _, H_f, W_f = filter_ref.shape
  _, _, K0, K1 = out_ref.shape

  feature_group_count = kwargs['feature_group_count']
  assert feature_group_count == C_in == C_out # TODO: relax me
  assert N == 1 # relax me
  kwargs['in_perm'] = [1, 0, 2, 3]

  MAX_F = 18200  # FIXME: Use nki.language.constants once max free size is added
  MAX_H = (MAX_F // sizeinbytes(img_ref.dtype)) // W
  H_NUM_TILES = H // MAX_H
  REMAINDER_H_TILE = H % MAX_H

  BGC_TILES, BGC_TILE_SIZE = div_ceil(feature_group_count, 128), min(feature_group_count, 128)
  kwargs['dsts_shapes'] = [[BGC_TILE_SIZE, BGC_TILE_SIZE, K0, K1]]
  for bgc_tile in nl.affine_range(BGC_TILES): # assumption on batch_group_count == N == C_out
    sparse_out_ref = nl.ndarray(shape=(BGC_TILE_SIZE, BGC_TILE_SIZE, K0, K1), dtype=dtype, buffer=nl.hbm, name='sparse_out')
    sub_img = img_ref
    sub_filter = filter_ref
    if BGC_TILES > 1:
      sub_img = nl.ndarray(shape=(BGC_TILE_SIZE, H, W), dtype=dtype, buffer=nl.hbm, name='sub_img')
      sub_filter = nl.ndarray(shape=(BGC_TILE_SIZE, N, H_f, W_f), dtype=dtype, buffer=nl.hbm, name='sub_filter')
      for h_tile in nl.affine_range(H_NUM_TILES):
        sub_img_sbuf = nl.load(img_ref[0, bgc_tile*BGC_TILE_SIZE:(bgc_tile+1)*BGC_TILE_SIZE, h_tile*MAX_H:(h_tile+1)*MAX_H, :])
        nl.store(sub_img[:, h_tile*MAX_H:(h_tile+1)*MAX_H, :], sub_img_sbuf)

      if REMAINDER_H_TILE > 0:
        sub_img_sbuf_remainder = nl.load(img_ref[0, bgc_tile*BGC_TILE_SIZE:(bgc_tile+1)*BGC_TILE_SIZE, H_NUM_TILES*MAX_H:H_NUM_TILES*MAX_H+REMAINDER_H_TILE, :])
        nl.store(sub_img[:, H_NUM_TILES*MAX_H:H_NUM_TILES*MAX_H+REMAINDER_H_TILE, :], sub_img_sbuf_remainder)

      sub_filter_sbuf = nl.load(filter_ref[bgc_tile*BGC_TILE_SIZE:(bgc_tile+1)*BGC_TILE_SIZE, :, :, :])
      nl.store(sub_filter[:, :, :, :], sub_filter_sbuf)
      kwargs['srcs_shapes'] = [[N, BGC_TILE_SIZE, H, W], [BGC_TILE_SIZE, N, H_f, W_f]]

    conv2d_dw_fb01_io01_01bf_rep_nhwc_Pcinh(sub_img, sub_filter, sparse_out_ref, **kwargs)
    # for out_idx in nl.affine_range(BGC_TILE_SIZE):
    #   # curr_window = nl.ndarray(shape=(1, K0*K1), dtype=dtype, buffer=nl.sbuf, name='curr_window')
    #   curr_window = nl.load(sparse_out_ref[out_idx*(BGC_TILE_SIZE+1):out_idx*(BGC_TILE_SIZE+1)+1, :])
    #   nl.store(out_ref[0:1, bgc_tile*BGC_TILE_SIZE + out_idx, :], curr_window)
    sparse_out_ref = sparse_out_ref.reshape([BGC_TILE_SIZE, BGC_TILE_SIZE, K0*K1])
    mask = nl.shared_constant(np.fromfunction(lambda i, j, k: (i == j).astype(np.uint8), (sparse_out_ref.shape), dtype=np.uint8))
    out_ref = out_ref.reshape([C_out, K0*K1]) # assuming N // batch_group_count == 1

    # load both inputs
    # sparse_input = nl.ndarray(shape=(BGC_TILE_SIZE, par_dim(BGC_TILE_SIZE), K0*K1), dtype=dtype, buffer=nl.sbuf, name='sparse_input')
    # mask_input = nl.ndarray(shape=(BGC_TILE_SIZE, par_dim(BGC_TILE_SIZE), K0*K1), dtype=dtype, buffer=nl.sbuf, name='mask_input')
    # for out_idx in nl.affine_range(BGC_TILE_SIZE):
    #   sparse_input[out_idx, :, :] = nl.load(sparse_out_ref[out_idx, :, :])
    #   mask_input[out_idx, :, :] = nl.load(mask[out_idx, :, :])
    # reduce in window of <BGC_TILE_SIZExMM_REDUCE_TILE_SIZE> -> <1xMM_REDUCE_TILE_SIZE>
    MM_REDUCE_TILES, MM_REDUCE_TILE_SIZE = div_ceil(K0*K1, 128), min(K0*K1, 128)
    mm_reduce_weight = nisa.memset((par_dim(BGC_TILE_SIZE), 1),
                                    value=1, dtype=dtype) # 

    reduced = nl.ndarray(shape=(MM_REDUCE_TILES, par_dim(MM_REDUCE_TILE_SIZE), BGC_TILE_SIZE), dtype=np.float32, buffer=nl.psum, name='reduced')
    # out_sb = nl.ndarray((MM_REDUCE_TILES, par_dim(BGC_TILE_SIZE), MM_REDUCE_TILE_SIZE), dtype=dtype, buffer=nl.sbuf, name=f'tranposed')
    
    
    for out_idx in nl.affine_range(BGC_TILE_SIZE):
      # apply mask
      l = nl.load(sparse_out_ref[out_idx, :, :])
      r = nl.load(mask[out_idx, :, :])
      masked = l * r
      
      for reduce_tile in nl.affine_range(MM_REDUCE_TILES):
        i_p = nl.arange(BGC_TILE_SIZE)[:, None]
        i_f = nl.arange(MM_REDUCE_TILE_SIZE)[None, :]
        i_img = reduce_tile*MM_REDUCE_TILE_SIZE+i_f
        reduced[reduce_tile, :, out_idx] \
          = nisa.nc_matmul(masked[i_p, i_img][i_img < K0*K1], mm_reduce_weight, 
                          is_stationary_onezero=True)
    for reduce_tile in nl.affine_range(MM_REDUCE_TILES):
      out_sb = nl.transpose(reduced[reduce_tile, :, :])
      i_p = nl.arange(BGC_TILE_SIZE)[:, None]
      i_f = nl.arange(MM_REDUCE_TILE_SIZE)[None, :]
      i_img = reduce_tile*MM_REDUCE_TILE_SIZE+i_f
      nl.store(out_ref[bgc_tile*BGC_TILE_SIZE+i_p, i_img], out_sb[i_p, i_f], mask=(i_img < K0*K1))

'''
    # reduced = nl.ndarray(shape=(K0*K1, BGC_TILE_SIZE), dtype=dtype, buffer=nl.sbuf, name='reduced')
    for out_idx in nl.affine_range(BGC_TILE_SIZE):
      l = nl.load(sparse_out_ref[out_idx, :, :])
      r = nl.load(mask[out_idx, :, :])
      masked = l * r
      reduced = nl.ndarray(shape=(1, K0*K1), dtype=dtype, buffer=nl.sbuf, name='reduced')
      # i_k0, i_k1 = nl.mgrid[0:K0, 0:K1]
      # reduced[0, i_k0 * K1 + i_k1] = nl.sum(masked, axis=0)
      # breakpoint()
      MM_REDUCE_TILES, MM_REDUCE_TILE_SIZE = div_ceil(K0*K1, 512), min(K0*K1, 512)
      mm_reduce_weight = nisa.memset((par_dim(BGC_TILE_SIZE), 1),
                                     value=1, dtype=dtype)
      for reduce_tile in nl.affine_range(MM_REDUCE_TILES):
        i_p, i_f = nl.mgrid[0:BGC_TILE_SIZE, 0:MM_REDUCE_TILE_SIZE]
        i_img = reduce_tile*MM_REDUCE_TILE_SIZE+i_f
        reduced_psum \
          = nisa.nc_matmul(mm_reduce_weight, masked[i_p, i_img],
                                      is_stationary_onezero=True, mask = (i_img < K0*K1))
        reduced[0, i_img] = nl.copy(reduced_psum[0, i_f], mask=(i_img < K0*K1))
      # reduced = nisa.nc_transpose(reduced)

      # out_ref = out_ref.reshape([K0*K1, 1, C_out]) # assuming N // batch_group_count == 1
        nl.store(out_ref[0, bgc_tile*BGC_TILE_SIZE+out_idx:bgc_tile*BGC_TILE_SIZE+out_idx+1, :], reduced[:, :])
'''
