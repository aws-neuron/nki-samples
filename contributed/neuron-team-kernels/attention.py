"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

Kernels written by the AWS Neuron.

WARNING: These kernels:
   - Are tested only against internal nightly builds
   - May not be compatible with public NeuronSDK releases
   - Have not been extensively tested across all input configurations
   - Carry no compatibility guarantees
   - The behavior of these kernels may be modified without prior notice

Attention kernels

"""
import numpy as np

from neuronxcc import nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.nccl as nccl
import neuronxcc.nki.language as nl
from attention_build_blocks import FlashConfig, _flash_attention_core, _flash_attn_bwd_core
from neuronxcc.nki.language import par_dim
from common import div_ceil
from collectives import CollectivesConfig

def fused_self_attn_bwd(
    q_ref, k_ref, v_ref,
    dy_ref,
    out_dq_ref, out_dk_ref, out_dv_ref,
    use_causal_mask=False,
    mixed_precision=False,
):
  """
  Fused self attention backward kernel. Compute the backward gradients.

  IO tensor layouts:
   - q_ref: shape (bs, nheads, head_size, seq)
   - k_ref: shape (bs, nheads, head_size, seq)
   - v_ref: shape (bs, nheads, head_size, seq)
   - dy_ref: shape (bs, nheads, head_size, seq)
   - out_dq_ref: shape (bs, nheads, head_size, seq)
   - out_dk_ref: shape (bs, nheads, head_size, seq)
   - out_dv_ref: shape (bs, nheads, head_size, seq)

  Detailed steps:
    1. Recompute (softmax(Q@K^T))
      1.1 Q@K^T
      1.2 Scale the QK score
      1.3 Apply causal mask
      1.4 softmax
    2. Compute the gradients of y = score @ V with respect to the loss

    3. Compute the gradients of y = softmax(x)

    4. Compute the gradients of Q@K^T
      4.1 Compute dQ
      4.2 Compute dK
  """

  # Use q_ref dtype as the intermediate tensor dtype
  # Assume all IO tensors have the same dtype
  kernel_dtype = q_ref.dtype
  tensor_acc_dtype = np.dtype(np.float32) if mixed_precision else kernel_dtype

  assert q_ref.dtype == k_ref.dtype == v_ref.dtype == dy_ref.dtype \
         == out_dq_ref.dtype == out_dk_ref.dtype == out_dv_ref.dtype

  # Shape checking
  bs, nheads, d_head, seqlen = q_ref.shape
  assert tuple(k_ref.shape) == (bs, nheads, d_head, seqlen), \
    f"Input K shape mismatch, got {k_ref.shape}"
  assert tuple(v_ref.shape) == (bs, nheads, d_head, seqlen), \
    f"Input V shape mismatch, got {v_ref.shape}"
  assert tuple(dy_ref.shape) == (bs, nheads, d_head, seqlen), \
    f"Input dy shape mismatch, got {dy_ref.shape}"

  assert tuple(out_dq_ref.shape) == (bs, nheads, d_head, seqlen), \
    f"Output dQ shape mismatch, got {out_dq_ref.shape}"
  assert tuple(out_dk_ref.shape) == (bs, nheads, d_head, seqlen), \
    f"Output dK shape mismatch, got {out_dk_ref.shape}"
  assert tuple(out_dv_ref.shape) == (bs, nheads, d_head, seqlen), \
    f"Output dV shape mismatch, got {out_dv_ref.shape}"

  # FIXME: Add masking for different seqlen values.
  assert seqlen % 128 == 0, \
    f"Input sequence length must be divisible by 128, got {seqlen}"

  # Softmax scaling factor, multiplied onto Q
  softmax_scale = 1.0 / float(d_head ** 0.5)

  # Different batch samples/attention heads have independent attention
  batch_id = nl.program_id(axis=0)
  head_id = nl.program_id(axis=1)

  q_seq_n_tiles, q_seq_tile_size = div_ceil(seqlen, 128), 128
  d_head_n_tiles, d_head_tile_size = div_ceil(d_head, 128), 128

  if seqlen >= 512:
    k_seq_n_tiles, k_seq_tile_size = seqlen // 512, 512
    v_seq_n_tiles, v_seq_tile_size = seqlen // 512, 512
  else:
    k_seq_n_tiles, k_seq_tile_size = seqlen // 128, 128
    v_seq_n_tiles, v_seq_tile_size = seqlen // 128, 128

  k_seq_n_tiles_backward, k_seq_tile_size_backward = seqlen // 128, 128
  k_seq_fwd_bwd_tile_multipler = k_seq_tile_size // k_seq_tile_size_backward

  ip_qk = nl.arange(d_head_tile_size)[:, None]
  if_q = nl.arange(q_seq_tile_size)[None, :]
  if_k = nl.arange(k_seq_tile_size)[None, :]

  # Prefetch dy

  # If head size is not a multiple of 128, we use 128 for tile size but mask out
  # computation as needed for numerical correctness. Note not all computation
  # is masked out, so we initialize relevant tensors to 0 to maintain numerical
  # correctness when head size is not a multiple of 128.
  dy_local = nl.zeros((q_seq_n_tiles, d_head_n_tiles, par_dim(d_head_tile_size), q_seq_tile_size), dtype=kernel_dtype)
  for i_d_head_tile in nl.affine_range(d_head_n_tiles):
    for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
      ip_qk_mask = i_d_head_tile * d_head_tile_size + ip_qk < d_head

      dy_local[i_q_seq_tile, i_d_head_tile, ip_qk, if_q] = nl.load(
        dy_ref[batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_qk, i_q_seq_tile * q_seq_tile_size + if_q],
        dtype=kernel_dtype,
        mask=ip_qk_mask)

  # Prefetch V
  v_local = nl.zeros((v_seq_n_tiles, d_head_n_tiles, par_dim(d_head_tile_size), v_seq_tile_size), dtype=kernel_dtype)
  for i_d_head_tile in nl.affine_range(d_head_n_tiles):
    for i_v_seq_tile in nl.affine_range(v_seq_n_tiles):
      ip_v = nl.arange(d_head_tile_size)[:, None]
      if_v = nl.arange(v_seq_tile_size)[None, :]

      ip_v_mask = i_d_head_tile * d_head_tile_size + ip_v < d_head

      v_local[i_v_seq_tile, i_d_head_tile, ip_v, if_v] = nl.load(
        v_ref[batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_v, i_v_seq_tile * v_seq_tile_size + if_v],
        dtype=kernel_dtype,
        mask=ip_v_mask)

  # Prefetch Q
  q_local = nl.zeros((q_seq_n_tiles, d_head_n_tiles, par_dim(d_head_tile_size), q_seq_tile_size), dtype=kernel_dtype)
  for i_d_head_tile in nl.affine_range(d_head_n_tiles):
    for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
      ##############################################################
      # Step 1.3 Scale the score. Here we multiply into q matrix directly,
      # which is mathematically equivalent
      ##############################################################
      ip_qk_mask = i_d_head_tile * d_head_tile_size + ip_qk < d_head

      q_local[i_q_seq_tile, i_d_head_tile, ip_qk, if_q] = nl.load(
        q_ref[batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_qk, i_q_seq_tile * q_seq_tile_size + if_q],
        dtype=kernel_dtype,
        mask=ip_qk_mask) * softmax_scale

  # Prefetch K
  k_local = nl.zeros((k_seq_n_tiles, d_head_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size), dtype=kernel_dtype)
  transposed_k_local = nl.zeros((k_seq_n_tiles_backward, d_head_n_tiles, par_dim(k_seq_tile_size_backward), d_head_tile_size), dtype=kernel_dtype)
  for i_d_head_tile in nl.affine_range(d_head_n_tiles):
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
      ip_qk_mask = i_d_head_tile * d_head_tile_size + ip_qk < d_head

      k_local[i_k_seq_tile, i_d_head_tile, ip_qk, if_k] = nl.load(
        k_ref[batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_qk, i_k_seq_tile * k_seq_tile_size + if_k],
        dtype=kernel_dtype,
        mask=ip_qk_mask)

      ##############################################################
      # Prefetch k transpose for the backward too
      ##############################################################
      if_k_backward = nl.arange(k_seq_tile_size_backward)[None, :]
      ip_k_backward = nl.arange(k_seq_tile_size_backward)[:, None]
      if_d_head = nl.arange(d_head_tile_size)[None, :]
      for i_k_seq_tile_backward in nl.affine_range(k_seq_fwd_bwd_tile_multipler):
        transposed_k_local[i_k_seq_tile * k_seq_fwd_bwd_tile_multipler + i_k_seq_tile_backward, i_d_head_tile, ip_k_backward, if_d_head] = \
          nisa.nc_transpose(k_local[i_k_seq_tile, i_d_head_tile, ip_qk,
                                    i_k_seq_tile_backward * k_seq_tile_size_backward + if_k_backward])


  dv_local_reduced = nl.zeros((k_seq_n_tiles, d_head_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size),
                                dtype=tensor_acc_dtype)
  dk_local_reduced = nl.zeros((k_seq_n_tiles, d_head_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size),
                                dtype=tensor_acc_dtype)
  for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
    # A SBUF buffer for an independent softmax tile
    qk_res_buf = nl.ndarray((par_dim(q_seq_tile_size), seqlen), buffer=nl.sbuf, dtype=kernel_dtype)
    neg_max_res = nl.full((par_dim(q_seq_tile_size), k_seq_n_tiles), fill_value=np.inf, buffer=nl.sbuf, dtype=kernel_dtype)
    # Loop over RHS free of matmul(stationary=tensor_q, moving=tensor_k, contract=d_head)
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
      forward_mask = i_q_seq_tile * q_seq_tile_size >= i_k_seq_tile * k_seq_tile_size if use_causal_mask else None
      # PSUM buffer shape: [q_seq_tile_size P, k_seq_tile_size F]
      qk_psum = nl.zeros((par_dim(q_seq_tile_size), k_seq_tile_size),
                         dtype=np.float32, buffer=nl.psum)

      # Tensor indices for accessing qk result in k_seq_tile_size
      ip_qk = nl.arange(q_seq_tile_size)[:, None]
      if_qk = nl.arange(k_seq_tile_size)[None, :]

      # Loop over contraction dim of QK matmul
      for i_d_head_tile in nl.affine_range(d_head_n_tiles):
        ##############################################################
        # Step 1.1 Compute Q@K^T, with matmul(stationary=tensor_q, moving=tensor_k, contract=d_head)
        ##############################################################
        qk_psum[ip_qk, if_qk] += nisa.nc_matmul(q_local[i_q_seq_tile, i_d_head_tile, ip_qk, if_q],
                                                k_local[i_k_seq_tile, i_d_head_tile, ip_qk, if_k],
                                                mask=forward_mask)

      ###################################
      # Step 1.2. Apply optional causal mask
      ###################################
      if use_causal_mask:
        # Magic number -9984.0 to replace -inf similar to what Tensorizer uses
        qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = nisa.affine_select(
          pred=(i_q_seq_tile * q_seq_tile_size + ip_qk >= i_k_seq_tile * k_seq_tile_size + if_qk),
          on_true_tile=qk_psum[ip_qk, if_qk], on_false_value=-9984.0, dtype=kernel_dtype,
          mask=forward_mask)
      else:
        # Simply send psum result back to sbuf
        qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = \
          nl.copy(qk_psum[ip_qk, if_qk], dtype=kernel_dtype)

      #######################################################
      # Step 1.4 Recompute the softmax in the forward
      #######################################################
      neg_max_res[ip_qk, i_k_seq_tile] = nisa.tensor_reduce(
        np.max, data=qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk],
        mask=forward_mask,
        axis=(1,), dtype=kernel_dtype, negate=True)

    if_max = nl.arange(k_seq_n_tiles)[None, :]
    neg_max_res_final = nl.ndarray((par_dim(q_seq_tile_size), 1), dtype=kernel_dtype)
    neg_max_res_final[ip_qk, 0] = nisa.tensor_reduce(
      np.min, data=neg_max_res[ip_qk, if_max],
      axis=(1,), dtype=kernel_dtype, negate=False)

    ip_softmax = nl.arange(q_seq_tile_size)[:, None]
    ip_sum_res = nl.arange(q_seq_tile_size)[:, None]

    softmax_numerator = nl.ndarray((par_dim(q_seq_tile_size), seqlen), dtype=kernel_dtype)
    sum_divisor = nl.ndarray((par_dim(q_seq_tile_size), 1), dtype=kernel_dtype)

    if_softmax = nl.arange(k_seq_tile_size)[None, :]
    sum_res_partial = nl.zeros((par_dim(q_seq_tile_size), k_seq_n_tiles), buffer=nl.sbuf, dtype=tensor_acc_dtype)
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
      forward_mask = i_q_seq_tile * q_seq_tile_size >= i_k_seq_tile * k_seq_tile_size if use_causal_mask else None
      softmax_numerator[ip_softmax, i_k_seq_tile * k_seq_tile_size + if_softmax] = \
          nisa.activation(np.exp,
                          data=qk_res_buf[ip_softmax, i_k_seq_tile * k_seq_tile_size + if_softmax],
                          bias=neg_max_res_final[ip_softmax, 0], scale=1.0,
                          mask=forward_mask)

      sum_res_partial[ip_softmax, i_k_seq_tile] = \
          nisa.tensor_reduce(np.add, data=softmax_numerator[ip_softmax, i_k_seq_tile * k_seq_tile_size + if_softmax],
                      axis=(1,), dtype=tensor_acc_dtype, mask=forward_mask)

    sum_res = nisa.tensor_reduce(np.add, data=sum_res_partial[ip_softmax, if_max], axis=(1,), dtype=tensor_acc_dtype)
    sum_reciprocal = 1.0 / sum_res
    sum_divisor[ip_sum_res, 0] = nl.copy(sum_reciprocal, dtype=kernel_dtype)

    softmax_y_times_dy_sum_partial = nl.zeros((par_dim(q_seq_tile_size), k_seq_n_tiles),
                                              dtype=tensor_acc_dtype, buffer=nl.sbuf)
    softmax_dy = nl.ndarray((par_dim(q_seq_tile_size), seqlen), dtype=kernel_dtype, buffer=nl.sbuf)
    softmax_y = nl.ndarray((par_dim(q_seq_tile_size), seqlen), dtype=kernel_dtype, buffer=nl.sbuf)
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
      ip_v = nl.arange(d_head_tile_size)[:, None]
      if_v = nl.arange(k_seq_tile_size)[None, :]
      backward_mask = i_q_seq_tile * q_seq_tile_size >= i_k_seq_tile * k_seq_tile_size if use_causal_mask else None
      softmax_y[ip_softmax, i_k_seq_tile * k_seq_tile_size + if_v] = \
          nl.multiply(softmax_numerator[ip_softmax, i_k_seq_tile * k_seq_tile_size + if_v],
                        sum_divisor[ip_softmax, 0],
                        mask=backward_mask)

      #####################################################################
      # Step 2.1 Calculate the backward gradients dL/dV, where y=softmax@V
      # in value projection with matmul(stationary=dy, moving=softmax)
      #####################################################################
      for i_d_head_tile in nl.affine_range(d_head_n_tiles):
        ip_dv = nl.arange(d_head_tile_size)[:, None]
        if_dv = nl.arange(k_seq_tile_size)[None, :]
        trans_dy = nisa.nc_transpose(dy_local[i_q_seq_tile, i_d_head_tile, ip_v, if_q],
                                     mask=backward_mask)
        dv_psum = nisa.nc_matmul(trans_dy,
                                 softmax_y[ip_softmax, i_k_seq_tile * k_seq_tile_size + if_dv],
                                 mask=backward_mask)

        ip_dv_mask = i_d_head_tile * d_head_tile_size + ip_dv < d_head

        dv_local_reduced[i_k_seq_tile, i_d_head_tile, ip_dv, if_dv] = nl.loop_reduce(
                      dv_psum, op=np.add, loop_indices=(i_q_seq_tile,),
                      dtype=tensor_acc_dtype,
                      mask=(backward_mask & ip_dv_mask if backward_mask else ip_dv_mask))


      #####################################################################
      # Step 2.2 Calculate the backward gradients dL/dsoftmax, where y=softmax@V
      # in value projection with matmul(stationary=dy, moving=v)
      #####################################################################
      softmax_dy_psum = nl.zeros((par_dim(q_seq_tile_size), k_seq_tile_size),
                                 dtype=np.float32, buffer=nl.psum)
      for i_d_head_tile in nl.affine_range(d_head_n_tiles):
        softmax_dy_psum[ip_softmax, if_v] += \
          nisa.nc_matmul(dy_local[i_q_seq_tile, i_d_head_tile, ip_v, if_q],
                         v_local[i_k_seq_tile, i_d_head_tile, ip_v, if_v],
                         mask=backward_mask)

      softmax_dy[ip_softmax, i_k_seq_tile * k_seq_tile_size + if_v] = \
        nl.copy(softmax_dy_psum[ip_softmax, if_v], dtype=kernel_dtype,
                  mask=backward_mask)

      #####################################################################
      # Step 3 Calculate the softmax backward gradients dL/dx, where y=softmax(x)
      # dL/dx = y * (dL/dy - sum(dL/dy * y)), where y = softmax(x)
      #####################################################################
      softmax_y_times_dy = nl.multiply(softmax_dy[ip_softmax, i_k_seq_tile * k_seq_tile_size + if_v],
                                       softmax_y[ip_softmax, i_k_seq_tile * k_seq_tile_size + if_v],
                                       dtype=kernel_dtype,
                                       mask=backward_mask)
      softmax_y_times_dy_sum_partial[ip_softmax, i_k_seq_tile] = \
        nisa.tensor_reduce(np.add, data=softmax_y_times_dy, axis=(1,), dtype=tensor_acc_dtype,
                    mask=backward_mask)

    softmax_y_times_dy_sum = nl.ndarray((par_dim(q_seq_tile_size), 1), dtype=tensor_acc_dtype)
    softmax_y_times_dy_sum[ip_softmax, 0] =  \
      nisa.tensor_reduce(np.add,
                  data=softmax_y_times_dy_sum_partial[ip_softmax, nl.arange(k_seq_n_tiles)[None, :]],
                  axis=(1, ), dtype=tensor_acc_dtype)

    if_k = nl.arange(k_seq_tile_size)[None, :]
    softmax_dx_local = nl.ndarray((k_seq_n_tiles, par_dim(q_seq_tile_size), k_seq_tile_size),
                                  dtype=kernel_dtype, buffer=nl.sbuf)
    transposed_softmax_dx_local = nl.ndarray((k_seq_n_tiles_backward, par_dim(k_seq_tile_size_backward), q_seq_tile_size),
                                             dtype=kernel_dtype, buffer=nl.sbuf)
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
      backward_mask = i_q_seq_tile * q_seq_tile_size >= i_k_seq_tile * k_seq_tile_size if use_causal_mask else None
      # y * (dL/dy - sum(dL/dy * y))
      softmax_dx_local[i_k_seq_tile, ip_softmax, if_k] = \
        nisa.scalar_tensor_tensor(data=softmax_dy[ip_softmax, i_k_seq_tile * k_seq_tile_size + if_k],
                                  op0=np.subtract,
                                  operand0=softmax_y_times_dy_sum[ip_softmax, 0],
                                  op1=np.multiply,
                                  operand1=softmax_y[ip_softmax, i_k_seq_tile * k_seq_tile_size + if_k],
                                  mask=backward_mask)

    #####################################################################
    # Step 4.2 Calculate dK, with matmul(stationary=Q, moving=softmax_dx)
    #####################################################################
    ip_trans_q = nl.arange(d_head_tile_size)[:, None]
    if_trans_q = nl.arange(q_seq_tile_size)[None, :]
    if_softmax_dx = nl.arange(k_seq_tile_size)[None, :]
    ip_dk = nl.arange(d_head_tile_size)[:, None]
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
        backward_mask = i_q_seq_tile * q_seq_tile_size >= i_k_seq_tile * k_seq_tile_size if use_causal_mask else None
        for i_d_head_tile in nl.affine_range(d_head_n_tiles):
          trans_q_local = nisa.nc_transpose(q_local[i_q_seq_tile, i_d_head_tile, ip_trans_q, if_trans_q],
                                            mask=backward_mask)
          dk_psum = nisa.nc_matmul(
                      trans_q_local,
                      softmax_dx_local[i_k_seq_tile, ip_softmax, if_softmax_dx],
                      mask=backward_mask)

          ip_dk_mask = i_d_head_tile * d_head_tile_size + ip_dk < d_head

          dk_local_reduced[i_k_seq_tile, i_d_head_tile, ip_dk, if_softmax_dx] = nl.loop_reduce(
            dk_psum, op=np.add, loop_indices=(i_q_seq_tile,),
            dtype=tensor_acc_dtype,
            mask=(backward_mask & ip_dk_mask if backward_mask else ip_dk_mask))

        # Transpose softmax_dx early to avoid the tranpose under contract dimension of dQ
        ip_k = nl.arange(k_seq_tile_size_backward)[:, None]
        if_k = nl.arange(k_seq_tile_size_backward)[None, :]
        for i_k_seq_tile_backward in nl.affine_range(k_seq_fwd_bwd_tile_multipler):
          transposed_softmax_dx_local[i_k_seq_tile * k_seq_fwd_bwd_tile_multipler + i_k_seq_tile_backward, ip_k, if_trans_q] = \
            nisa.nc_transpose(softmax_dx_local[i_k_seq_tile, ip_softmax,
                                               i_k_seq_tile_backward * k_seq_tile_size_backward + if_k],
                              mask=backward_mask)

    #####################################################################
    # Step 4.1 Calculate dQ
    #####################################################################
    ip_k = nl.arange(d_head_tile_size)[:, None]
    if_k = nl.arange(k_seq_tile_size_backward)[None, :]
    ip_dq = nl.arange(d_head_tile_size)[:, None]
    if_dq = nl.arange(q_seq_tile_size)[None, :]
    ip_transposed_k = nl.arange(k_seq_tile_size_backward)[:, None]
    for i_d_head_tile in nl.affine_range(d_head_n_tiles):
      dq_psum = nl.zeros((par_dim(d_head_tile_size), q_seq_tile_size),
                         dtype=np.float32, buffer=nl.psum)
      for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
        backward_mask = i_q_seq_tile * q_seq_tile_size >= i_k_seq_tile * k_seq_tile_size if use_causal_mask else None
        for i_k_seq_tile_backward in nl.affine_range(k_seq_fwd_bwd_tile_multipler):
          dq_psum[ip_dq, if_dq] += nisa.nc_matmul(transposed_k_local[i_k_seq_tile * k_seq_fwd_bwd_tile_multipler + i_k_seq_tile_backward,
                                                                     i_d_head_tile, ip_transposed_k, if_dq],
                                                  transposed_softmax_dx_local[i_k_seq_tile * k_seq_fwd_bwd_tile_multipler + i_k_seq_tile_backward,
                                                                              ip_transposed_k, if_dq],
                                                  mask=backward_mask)

      dq_local = nl.multiply(dq_psum[ip_dq, if_dq], softmax_scale, dtype=kernel_dtype)

      ip_dq_mask = i_d_head_tile * d_head_tile_size + ip_dq < d_head

      nl.store(
        out_dq_ref[batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_dq, i_q_seq_tile * q_seq_tile_size + if_dq],
        value=dq_local,
        mask=ip_dq_mask
      )

  #####################################################################
  # Store dK, dV (at end to maintain loop fusion)
  #####################################################################
  for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
    for i_d_head_tile in nl.affine_range(d_head_n_tiles):
      ip_dkv = nl.arange(d_head_tile_size)[:, None]
      if_dkv = nl.arange(k_seq_tile_size)[None, :]


      ip_dkv_mask = i_d_head_tile * d_head_tile_size + ip_dkv < d_head

      nl.store(
        out_dv_ref[batch_id, head_id,
                    i_d_head_tile * d_head_tile_size + ip_dv,
                    i_k_seq_tile * k_seq_tile_size + if_dv],
        value=dv_local_reduced[i_k_seq_tile, i_d_head_tile, ip_dkv, if_dkv],
        mask=ip_dkv_mask
      )

      nl.store(
        out_dk_ref[batch_id, head_id,
                    i_d_head_tile * d_head_tile_size + ip_dk,
                    i_k_seq_tile * k_seq_tile_size + if_softmax_dx],
        value=dk_local_reduced[i_k_seq_tile, i_d_head_tile, ip_dkv, if_dkv],
        mask=ip_dkv_mask
      )

def global_rank_to_local_rank(global_replica_group, global_rank):
  """Convert a global rank to the local rank according to the global_replica_group
  For example, if the global_replica_group is [0, 8, 16, 24],
  the local rank id mapping will be {0->0, 8->1, 16->2, 24->3}.

  Args:
      global_replica_group (_type_): list of global ranks in the replica group
      global_rank_id (_type_): current global rank

  Returns:
      _type_: current local rank in the replica group
  """
  global_ranks = sorted(global_replica_group)
  local_rank = global_ranks.index(global_rank)
  return local_rank

def get_global_ring_order(src_tgt_pairs, global_rank):
  """Get the ring order in global ranks starting from the current global_rank
  represented as a "receive from" ring,
  i.e. global_ring[i] is receiving from global_ring[i+1]

  Args:
      src_tgt_pairs (_type_): list of pairs, in no particular order
      global_rank (_type_): the current global rank
  """
  recv_from_graph = {}
  for pair in src_tgt_pairs:
    src_global_rank, tgt_global_rank = pair
    recv_from_graph[tgt_global_rank] = src_global_rank
  global_ring = [global_rank]
  while True:
    curr_global_rank = global_ring[-1]
    global_rank_from = recv_from_graph[curr_global_rank]
    if global_rank_from in global_ring:
      break
    else:
      global_ring.append(global_rank_from)
  assert len(global_ring) == len(src_tgt_pairs), f"src_tgt_pairs {src_tgt_pairs} does not form a closed simple ring."
  return global_ring

def global_ring_to_local_ring(global_ring_order):
  """Return a corresponding local recv order from the global_ring_order
  For example, if the global_ring_order is [0, 8, 16, 24],
  the local rank id mapping will be {0->0, 8->1, 16->2, 24->3}.
  local_ring_order will be [3, 2, 1, 0]
  ring attention performs ring size - 1 communications,
  so a rank will receive from all ranks except itself
  """
  local_ring_order = [global_rank_to_local_rank(global_ring_order, global_rank) for global_rank in global_ring_order]
  return local_ring_order

@nki.jit
def ring_attention_bwd(
  q_ref, k_ref, v_ref, o_ref,
  dy_ref,
  lse_ref,
  seed_ref,
  rank_id=0,
  src_tgt_pairs=[],
  use_causal_mask=False,
  mixed_precision=False,
  dropout_p=0.0,
  softmax_scale=None,
):
  """
  Ring attention backward kernel. Compute the backward gradients with distributed inputs.

  IO tensor layouts:
   - q_ref: shape (bs, nheads, head_size, seq)
   - k_ref: shape (bs, nheads, head_size, seq)
   - v_ref: shape (bs, nheads, head_size, seq)
   - o_ref: shape (bs, nheads, head_size, seq)
   - dy_ref: shape (bs, nheads, head_size, seq)
   - lse_ref: shape (bs, nheads, nl.tile_size.pmax, seq // nl.tile_size.pmax)
   - out_dq_ref: shape (bs, nheads, head_size, seq)
   - out_dk_ref: shape (bs, nheads, head_size, seq)
   - out_dv_ref: shape (bs, nheads, head_size, seq)

  """

  # Use q_ref dtype as the intermediate tensor dtype
  # Assume all IO tensors have the same dtype
  kernel_dtype = q_ref.dtype
  mixed_dtype = np.dtype(np.float32) if mixed_precision else kernel_dtype
  
  cc_configs = CollectivesConfig(src_tgt_pairs=src_tgt_pairs)
  rank_src_tgt_pairs = cc_configs.rank_src_tgt_pairs[rank_id]
  global_ring_order = cc_configs.rank_recv_replica_groups[rank_id]
  
  num_workers = max(len(rank_src_tgt_pairs),1)
  local_rank_id = global_rank_to_local_rank(global_ring_order, rank_id)
  local_ring_order = global_ring_to_local_ring(global_ring_order)
  
  out_dq_ref = nl.ndarray(q_ref.shape, dtype=q_ref.dtype, buffer=nl.shared_hbm)
  out_dk_ref = nl.ndarray(q_ref.shape, dtype=q_ref.dtype, buffer=nl.shared_hbm)
  out_dv_ref = nl.ndarray(q_ref.shape, dtype=q_ref.dtype, buffer=nl.shared_hbm)

  assert q_ref.dtype == k_ref.dtype == v_ref.dtype == o_ref.dtype == dy_ref.dtype \
         == out_dq_ref.dtype == out_dk_ref.dtype == out_dv_ref.dtype
  assert lse_ref.dtype == mixed_dtype

  # Shape checking
  bs, nheads, d_head, seqlen = q_ref.shape
  assert tuple(k_ref.shape) == (bs, nheads, d_head, seqlen), \
    f"Input K shape mismatch, got {k_ref.shape}"
  assert tuple(v_ref.shape) == (bs, nheads, d_head, seqlen), \
    f"Input V shape mismatch, got {v_ref.shape}"
  assert tuple(o_ref.shape) == (bs, nheads, d_head, seqlen), \
    f"Input dy shape mismatch, got {o_ref.shape}"
  assert tuple(dy_ref.shape) == (bs, nheads, d_head, seqlen), \
    f"Input dy shape mismatch, got {dy_ref.shape}"
  assert tuple(lse_ref.shape) == (bs, nheads, nl.tile_size.pmax, seqlen // nl.tile_size.pmax), \
    f"Input lse shape mismatch, got {lse_ref.shape}"
  if seed_ref is not None:
    assert tuple(seed_ref.shape) == (1,), \
      f"Input seed shape mismatch, got {seed_ref.shape}"

  assert tuple(out_dq_ref.shape) == (bs, nheads, d_head, seqlen), \
    f"Output dQ shape mismatch, got {out_dq_ref.shape}"
  assert tuple(out_dk_ref.shape) == (bs, nheads, d_head, seqlen), \
    f"Output dK shape mismatch, got {out_dk_ref.shape}"
  assert tuple(out_dv_ref.shape) == (bs, nheads, d_head, seqlen), \
    f"Output dV shape mismatch, got {out_dv_ref.shape}"

  # FIXME: Add masking for different seqlen values.
  assert seqlen % 128 == 0, \
    f"Input sequence length must be divisible by 128, got {seqlen}"

  # Softmax scaling factor, multiplied onto Q
  softmax_scale = softmax_scale or 1.0 / float(d_head ** 0.5)

  # Different batch samples/attention heads have independent attention
  batch_id = nl.program_id(axis=0)
  head_id = nl.program_id(axis=1)

  q_seq_n_tiles, q_seq_tile_size = div_ceil(seqlen, 128), 128
  d_head_n_tiles, d_head_tile_size = div_ceil(d_head, 128), min(d_head, 128)

  if seqlen >= 512:
    k_seq_n_tiles, k_seq_tile_size = seqlen // 512, 512
  else:
    k_seq_n_tiles, k_seq_tile_size = seqlen // 128, 128

  k_seq_n_tiles_backward, k_seq_tile_size_backward = seqlen // 128, 128
  k_seq_fwd_bwd_tile_multipler = k_seq_tile_size // k_seq_tile_size_backward

  ##############################################################
  # Step 2.4 Prefetch exp bias for softmax
  ##############################################################
  softmax_exp_bias = nl.zeros((q_seq_n_tiles, par_dim(q_seq_tile_size), 1), dtype=mixed_dtype)
  for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
    ip_qk = nl.arange(q_seq_tile_size)[:, None]
    lse_local = nl.load(
      lse_ref[batch_id, head_id, ip_qk, i_q_seq_tile],
      dtype=mixed_dtype)
    softmax_exp_bias[i_q_seq_tile, ip_qk, 0] = lse_local * -1.0

  ##############################################################
  # Step 1 Compute rowsum(dO ◦ O)
  ##############################################################
  dy_o_sum = nl.ndarray((q_seq_n_tiles, par_dim(q_seq_tile_size), 1), dtype=mixed_dtype)
  for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
    ip_reduce = nl.arange(q_seq_tile_size)[:, None]
    dy_o_partial = nl.zeros((par_dim(q_seq_tile_size), d_head_n_tiles), dtype=mixed_dtype)
    for i_d_head_tile in nl.affine_range(d_head_n_tiles):
      ip_load = nl.arange(d_head_tile_size)[:, None]
      if_q = nl.arange(q_seq_tile_size)[None, :]
      dy_local = nl.load_transpose2d(
        dy_ref[batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_load, i_q_seq_tile * q_seq_tile_size + if_q],
        dtype=mixed_dtype)
      o_local = nl.load_transpose2d(
        o_ref[batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_load, i_q_seq_tile * q_seq_tile_size + if_q],
        dtype=mixed_dtype
      )

      dy_o_partial[ip_reduce, i_d_head_tile] = nisa.tensor_reduce(
        np.add, data=dy_local*o_local, axis=(1,), dtype=mixed_dtype
      )

    dy_o_sum[i_q_seq_tile, ip_reduce, 0] = nisa.tensor_reduce(
      np.add, data=dy_o_partial[ip_reduce, nl.arange(d_head_n_tiles)[None, :]],
      axis=(1,), dtype=mixed_dtype
    )

  # Indices for prefetch
  ip_qk = nl.arange(d_head_tile_size)[:, None]
  if_q = nl.arange(q_seq_tile_size)[None, :]
  if_k = nl.arange(k_seq_tile_size)[None, :]

  if dropout_p > 0.0:
    seed_local = nl.ndarray((par_dim(1), 1), buffer=nl.sbuf, dtype=nl.int32)
    seed_local[0, 0] = nl.load(seed_ref[0])
    # TODO: Remove this once the dropout supports scale prob
    dropout_p_local = nl.full((q_seq_tile_size, 1), fill_value=dropout_p, dtype=np.float32)
  else:
    seed_local = None
    dropout_p_local = None

  dq_local_reduced = nl.zeros((q_seq_n_tiles, d_head_n_tiles, par_dim(d_head_tile_size), q_seq_tile_size),
                              dtype=mixed_dtype)

  # Local buffer to hold dK and dV
  dk_acc_buf = nl.ndarray((d_head, seqlen), dtype=nl.float32, buffer=nl.private_hbm, name="dk_acc_buf")
  dv_acc_buf = nl.ndarray((d_head, seqlen), dtype=nl.float32, buffer=nl.private_hbm, name="dv_acc_buf")

  # Double buffer to hold the Q, dy and dy_o_sum
  send_q_buf = nl.ndarray((d_head, seqlen), dtype=q_ref.dtype, buffer=nl.private_hbm, name="send_q_buf")
  recv_q_buf = nl.ndarray((d_head, seqlen), dtype=q_ref.dtype, buffer=nl.private_hbm, name="recv_q_buf")
  send_dy_buf = nl.ndarray((d_head, seqlen), dtype=dy_ref.dtype, buffer=nl.private_hbm, name="send_dy_buf")
  recv_dy_buf = nl.ndarray((d_head, seqlen), dtype=dy_ref.dtype, buffer=nl.private_hbm, name="recv_dy_buf")
  send_lse_buf = nl.ndarray((nl.tile_size.pmax, seqlen // nl.tile_size.pmax), dtype=lse_ref.dtype, buffer=nl.private_hbm, name="send_lse_buf")
  recv_lse_buf = nl.ndarray((nl.tile_size.pmax, seqlen // nl.tile_size.pmax), dtype=lse_ref.dtype, buffer=nl.private_hbm, name="send_lse_buf")
  send_dy_o_sum_buf = nl.ndarray((seqlen, 1), dtype=dy_o_sum.dtype, buffer=nl.private_hbm, name="send_dy_o_sum_buf")
  recv_dy_o_sum_buf = nl.ndarray((seqlen, 1), dtype=dy_o_sum.dtype, buffer=nl.private_hbm, name="recv_dy_o_sum_buf")
  send_dq_buf = nl.ndarray((d_head, seqlen), dtype=nl.float32, buffer=nl.private_hbm, name="send_dq_buf")
  recv_dq_buf = nl.ndarray((d_head, seqlen), dtype=nl.float32, buffer=nl.private_hbm, name="recv_dq_buf")

  ip_send_q_buf, if_send_q_buf = nl.mgrid[0:d_head_tile_size, 0:seqlen]
  ip_dy_o_sum_buf, if_dy_o_sum_buf = nl.mgrid[0:q_seq_tile_size, 0:1]
  ip_lse_buf, if_lse_buf = nl.mgrid[0:nl.tile_size.pmax, 0:(seqlen // nl.tile_size.pmax)]

  # Initialize the buffer
  for i_d_head_tile in nl.affine_range(d_head_n_tiles):
    nisa._tiled_offloaded_memcpy(dst=send_q_buf[i_d_head_tile * d_head_tile_size + ip_send_q_buf, if_send_q_buf],
                                 src=q_ref[batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_send_q_buf, if_send_q_buf])
    nisa._tiled_offloaded_memcpy(dst=send_dy_buf[i_d_head_tile * d_head_tile_size + ip_send_q_buf, if_send_q_buf],
                                 src=dy_ref[batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_send_q_buf, if_send_q_buf])

  for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
    nisa._tiled_offloaded_memcpy(dst=send_lse_buf[ip_lse_buf, if_lse_buf],
                                 src=lse_ref[batch_id, head_id, ip_lse_buf, if_lse_buf])

    nl.store(dst=send_dy_o_sum_buf[i_q_seq_tile * q_seq_tile_size + ip_dy_o_sum_buf, if_dy_o_sum_buf],
             value=dy_o_sum[i_q_seq_tile, ip_dy_o_sum_buf, if_dy_o_sum_buf])

  # Send the local buffers to the neighbors
  def _collective_permute_buffers():
    # for i_d_head_tile in nl.affine_range(d_head_n_tiles):
    nccl.collective_permute(dst=recv_q_buf[:, :], src=send_q_buf[:, :],
                            source_target_pairs=src_tgt_pairs)
    nccl.collective_permute(dst=recv_dy_buf[:, :], src=send_dy_buf[:, :],
                            source_target_pairs=src_tgt_pairs)
    nccl.collective_permute(dst=recv_dy_o_sum_buf[:, :], src=send_dy_o_sum_buf[:, :],
                            source_target_pairs=src_tgt_pairs)
    nccl.collective_permute(dst=recv_lse_buf[:, :], src=send_lse_buf[:, :],
                            source_target_pairs=src_tgt_pairs)
  _collective_permute_buffers()

  # affine_range gives the compiler permission to vectorize instructions
  # inside the loop which improves the performance. However, when using the
  # the dropout we should use sequential_range to avoid setting
  # seed vectorization. TODO: the compiler should avoid vectorizing seed setting
  _range = nl.sequential_range if dropout_p > 0.0 else nl.affine_range

  # Calculate the gradients based on the local Q and dy
  for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
    # Prefetch V, K
    v_local = nl.zeros((d_head_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size), dtype=kernel_dtype)
    k_local = nl.zeros((d_head_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size), dtype=kernel_dtype)
    transposed_k_local = nl.zeros((k_seq_fwd_bwd_tile_multipler, d_head_n_tiles, par_dim(k_seq_tile_size_backward), d_head_tile_size), dtype=kernel_dtype)
    for i_d_head_tile in nl.affine_range(d_head_n_tiles):
      k_local[i_d_head_tile, ip_qk, if_k] = nl.load(
        k_ref[batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_qk, i_k_seq_tile * k_seq_tile_size + if_k],
        dtype=kernel_dtype)
      v_local[i_d_head_tile, ip_qk, if_k] = nl.load(
        v_ref[batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_qk, i_k_seq_tile * k_seq_tile_size + if_k],
        dtype=kernel_dtype)
      ##############################################################
      # Prefetch k transpose for the backward too
      ##############################################################
      if_k_backward = nl.arange(k_seq_tile_size_backward)[None, :]
      ip_k_backward = nl.arange(k_seq_tile_size_backward)[:, None]
      if_d_head = nl.arange(d_head_tile_size)[None, :]
      for i_k_seq_tile_backward in nl.affine_range(k_seq_fwd_bwd_tile_multipler):
        transposed_k_local[i_k_seq_tile_backward, i_d_head_tile, ip_k_backward, if_d_head] = \
          nisa.nc_transpose(k_local[i_d_head_tile, ip_qk,
                                    i_k_seq_tile_backward * k_seq_tile_size_backward + if_k_backward])

    dv_psum = nl.zeros((d_head_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size),
                        dtype=np.float32, buffer=nl.psum)
    dk_psum = nl.zeros((d_head_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size),
                        dtype=np.float32, buffer=nl.psum)
    for i_q_seq_tile in _range(q_seq_n_tiles):
      # Prefetch dy, Q
      dy_local = nl.zeros((d_head_n_tiles, par_dim(d_head_tile_size), q_seq_tile_size), dtype=kernel_dtype)
      q_local = nl.zeros((d_head_n_tiles, par_dim(d_head_tile_size), q_seq_tile_size), dtype=kernel_dtype)
      for i_d_head_tile in nl.affine_range(d_head_n_tiles):
        ip_qk = nl.arange(d_head_tile_size)[:, None]
        if_q = nl.arange(q_seq_tile_size)[None, :]

        dy_local[i_d_head_tile, ip_qk, if_q] = nl.load(
          dy_ref[batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_qk, i_q_seq_tile * q_seq_tile_size + if_q],
          dtype=kernel_dtype)

        q_local[i_d_head_tile, ip_qk, if_q] = nl.load(
          q_ref[batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_qk, i_q_seq_tile * q_seq_tile_size + if_q],
          dtype=kernel_dtype) * softmax_scale

      _flash_attn_bwd_core(
        q_local=q_local, k_local=k_local, transposed_k_local=transposed_k_local,
        v_local=v_local, dy_local=dy_local,
        dk_psum=dk_psum, dv_psum=dv_psum, dq_local_reduced=dq_local_reduced,
        softmax_exp_bias=softmax_exp_bias, dy_o_sum=dy_o_sum,
        local_i_q_seq_tile=i_q_seq_tile, local_i_k_seq_tile=i_k_seq_tile,
        global_i_q_seq_tile=local_rank_id * q_seq_n_tiles + i_q_seq_tile,
        global_i_k_seq_tile=local_rank_id * k_seq_n_tiles + i_k_seq_tile,
        seqlen_q=seqlen, seqlen_k=seqlen, d_head=d_head, # TODO: upgrade this kernel to handle non-square seqlens
        use_causal_mask=use_causal_mask,
        kernel_dtype=kernel_dtype, mixed_dtype=mixed_dtype,
        softmax_scale=softmax_scale,
        seed_local=seed_local, dropout_p=dropout_p, dropout_p_local=dropout_p_local,
      )

    # Write dK, dV
    for i_d_head_tile in nl.affine_range(d_head_n_tiles):
      ip_dkv = nl.arange(d_head_tile_size)[:, None]
      if_dkv = nl.arange(k_seq_tile_size)[None, :]

      nl.store(
        dv_acc_buf[i_d_head_tile * d_head_tile_size + ip_dkv,
                   i_k_seq_tile * k_seq_tile_size + if_dkv],
        value=dv_psum[i_d_head_tile, ip_dkv, if_dkv],
      )

      nl.store(
        dk_acc_buf[i_d_head_tile * d_head_tile_size + ip_dkv,
                   i_k_seq_tile * k_seq_tile_size + if_dkv],
        value=dk_psum[i_d_head_tile, ip_dkv, if_dkv],
      )

  # Write dq to local buffer and send it to next neighbor
  for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
    for i_d_head_tile in nl.affine_range(d_head_n_tiles):
      ip_dq = nl.arange(d_head_tile_size)[:, None]
      if_dq = nl.arange(q_seq_tile_size)[None, :]

      nl.store(
        send_dq_buf[i_d_head_tile * d_head_tile_size + ip_dq,
                    i_q_seq_tile * q_seq_tile_size + if_dq],
        value=dq_local_reduced[i_q_seq_tile, i_d_head_tile, ip_dq, if_dq],
      )

  nccl.collective_permute(dst=recv_dq_buf[:, :], src=send_dq_buf[:, :],
                          source_target_pairs=src_tgt_pairs)
  # Swap the buffer
  def _swap_buffer():
    nisa._tiled_offloaded_memcpy(dst=send_q_buf[:, :], src=recv_q_buf[:, :])
    nisa._tiled_offloaded_memcpy(dst=send_dy_buf[:, :], src=recv_dy_buf[:, :])
    nisa._tiled_offloaded_memcpy(dst=send_dy_o_sum_buf[:, :], src=recv_dy_o_sum_buf[:, :])
    nisa._tiled_offloaded_memcpy(dst=send_lse_buf[:, :], src=recv_lse_buf[:, :])

    for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
      dy_o_sum[i_q_seq_tile, ip_dy_o_sum_buf, if_dy_o_sum_buf] = \
        nl.load(send_dy_o_sum_buf[i_q_seq_tile * q_seq_tile_size + ip_dy_o_sum_buf, if_dy_o_sum_buf])

  _swap_buffer()

  # Keep receiving the q, dy from neighbors
  # TODO: Use sequential_range
  for ring_step in nl.static_range(1, num_workers):
    recv_local_rank = local_ring_order[ring_step]
    dk_buf = nl.ndarray((d_head, seqlen), dtype=nl.float32, buffer=nl.private_hbm, name="dk_buf")
    dv_buf = nl.ndarray((d_head, seqlen), dtype=nl.float32, buffer=nl.private_hbm, name="dv_buf")
    dq_local_reduced = nl.zeros((q_seq_n_tiles, d_head_n_tiles, par_dim(d_head_tile_size), q_seq_tile_size),
                                dtype=mixed_dtype)

    _collective_permute_buffers()
    # Prefetch lse
    for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
      lse_local = nl.load(
        send_lse_buf[ip_qk, i_q_seq_tile],
        dtype=mixed_dtype)
      softmax_exp_bias[i_q_seq_tile, ip_qk, 0] = lse_local * -1.0

    # Calculate the gradients based on the local Q and dy
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
      # Prefetch V, K
      v_local = nl.zeros((d_head_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size), dtype=kernel_dtype)
      k_local = nl.zeros((d_head_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size), dtype=kernel_dtype)
      transposed_k_local = nl.zeros((k_seq_fwd_bwd_tile_multipler, d_head_n_tiles, par_dim(k_seq_tile_size_backward), d_head_tile_size), dtype=kernel_dtype)
      for i_d_head_tile in nl.affine_range(d_head_n_tiles):
        k_local[i_d_head_tile, ip_qk, if_k] = nl.load(
          k_ref[batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_qk, i_k_seq_tile * k_seq_tile_size + if_k],
          dtype=kernel_dtype)
        v_local[i_d_head_tile, ip_qk, if_k] = nl.load(
          v_ref[batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_qk, i_k_seq_tile * k_seq_tile_size + if_k],
          dtype=kernel_dtype)
        ##############################################################
        # Prefetch k transpose for the backward too
        ##############################################################
        if_k_backward = nl.arange(k_seq_tile_size_backward)[None, :]
        ip_k_backward = nl.arange(k_seq_tile_size_backward)[:, None]
        if_d_head = nl.arange(d_head_tile_size)[None, :]
        for i_k_seq_tile_backward in nl.affine_range(k_seq_fwd_bwd_tile_multipler):
          transposed_k_local[i_k_seq_tile_backward, i_d_head_tile, ip_k_backward, if_d_head] = \
            nisa.nc_transpose(k_local[i_d_head_tile, ip_qk,
                                      i_k_seq_tile_backward * k_seq_tile_size_backward + if_k_backward])

      dv_psum = nl.zeros((d_head_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size),
                          dtype=np.float32, buffer=nl.psum)
      dk_psum = nl.zeros((d_head_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size),
                          dtype=np.float32, buffer=nl.psum)
      for i_q_seq_tile in _range(q_seq_n_tiles):
        # Prefetch dy, Q
        dy_local = nl.zeros((d_head_n_tiles, par_dim(d_head_tile_size), q_seq_tile_size), dtype=kernel_dtype)
        q_local = nl.zeros((d_head_n_tiles, par_dim(d_head_tile_size), q_seq_tile_size), dtype=kernel_dtype)
        for i_d_head_tile in nl.affine_range(d_head_n_tiles):
          ip_qk = nl.arange(d_head_tile_size)[:, None]
          if_q = nl.arange(q_seq_tile_size)[None, :]

          dy_local[i_d_head_tile, ip_qk, if_q] = nl.load(
            send_dy_buf[i_d_head_tile * d_head_tile_size + ip_qk, i_q_seq_tile * q_seq_tile_size + if_q],
            dtype=kernel_dtype)

          q_local[i_d_head_tile, ip_qk, if_q] = nl.load(
            send_q_buf[i_d_head_tile * d_head_tile_size + ip_qk, i_q_seq_tile * q_seq_tile_size + if_q],
            dtype=kernel_dtype) * softmax_scale

        _flash_attn_bwd_core(
          q_local=q_local, k_local=k_local, transposed_k_local=transposed_k_local,
          v_local=v_local, dy_local=dy_local,
          dk_psum=dk_psum, dv_psum=dv_psum, dq_local_reduced=dq_local_reduced,
          softmax_exp_bias=softmax_exp_bias, dy_o_sum=dy_o_sum,
          local_i_q_seq_tile=i_q_seq_tile, local_i_k_seq_tile=i_k_seq_tile,
          global_i_q_seq_tile=recv_local_rank * q_seq_n_tiles + i_q_seq_tile,
          global_i_k_seq_tile=local_rank_id * k_seq_n_tiles + i_k_seq_tile,
          seqlen_q=seqlen, seqlen_k=seqlen, d_head=d_head, # TODO: upgrade this kernel to handle non-square seqlens
          use_causal_mask=use_causal_mask,
          kernel_dtype=kernel_dtype, mixed_dtype=mixed_dtype,
          softmax_scale=softmax_scale,
          seed_local=seed_local, dropout_p=dropout_p, dropout_p_local=dropout_p_local,
        )

      # Write dK, dV
      for i_d_head_tile in nl.affine_range(d_head_n_tiles):
        ip_dkv = nl.arange(d_head_tile_size)[:, None]
        if_dkv = nl.arange(k_seq_tile_size)[None, :]

        nl.store(
          dv_buf[i_d_head_tile * d_head_tile_size + ip_dkv,
                 i_k_seq_tile * k_seq_tile_size + if_dkv],
          value=dv_psum[i_d_head_tile, ip_dkv, if_dkv],
        )

        nl.store(
          dk_buf[i_d_head_tile * d_head_tile_size + ip_dkv,
                 i_k_seq_tile * k_seq_tile_size + if_dkv],
          value=dk_psum[i_d_head_tile, ip_dkv, if_dkv],
        )


    # Write dq to local buffer and send it to next neighbor
    dq_add_tmp_buf = nl.ndarray((d_head, seqlen), dtype=nl.float32, buffer=nl.private_hbm, name="dq_add_tmp_buf")
    for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
      for i_d_head_tile in nl.affine_range(d_head_n_tiles):
        ip_dq = nl.arange(d_head_tile_size)[:, None]
        if_dq = nl.arange(q_seq_tile_size)[None, :]

        nl.store(
          dq_add_tmp_buf[i_d_head_tile * d_head_tile_size + ip_dq,
                         i_q_seq_tile * q_seq_tile_size + if_dq],
          value=dq_local_reduced[i_q_seq_tile, i_d_head_tile, ip_dq, if_dq],
        )

    nisa._tiled_offloaded_fma(
      dq_add_tmp_buf[:, :], recv_dq_buf[:, :],
      scales=[1.0, 1.0],
      dst=send_dq_buf[:, :]
    )
    nccl.collective_permute(dst=recv_dq_buf[:, :], src=send_dq_buf[:, :],
                            source_target_pairs=src_tgt_pairs)


    dk_fma_tmp_buf = nl.ndarray((d_head, seqlen), dtype=nl.float32, buffer=nl.private_hbm, name="dk_fma_tmp_buf")
    dv_fma_tmp_buf = nl.ndarray((d_head, seqlen), dtype=nl.float32, buffer=nl.private_hbm, name="dv_fma_tmp_buf")
    # Accumulate the dK dV results
    nisa._tiled_offloaded_fma(
      dk_buf[:, :], dk_acc_buf[:, :],
      scales=[1.0, 1.0],
      dst=dk_fma_tmp_buf[:, :],
    )
    nisa._tiled_offloaded_memcpy(
      dst=dk_acc_buf[:, :], src=dk_fma_tmp_buf[:, :]
    )
    nisa._tiled_offloaded_fma(
      dv_buf[:, :], dv_acc_buf[:, :],
      scales=[1.0, 1.0],
      dst=dv_fma_tmp_buf[:, :],
    )
    nisa._tiled_offloaded_memcpy(
      dst=dv_acc_buf[:, :], src=dv_fma_tmp_buf[:, :]
    )

    # Swap the buffer
    _swap_buffer()

  # Write to final output dK, dV
  for i_d_head_tile in nl.affine_range(d_head_n_tiles):
    ip_dkv = nl.arange(d_head_tile_size)[:, None]
    if_seq = nl.arange(seqlen)[None, :]

    nisa._tiled_offloaded_memcpy(
      dst=out_dv_ref[batch_id, head_id,
                     i_d_head_tile * d_head_tile_size + ip_dkv, if_seq],
      src=dv_acc_buf[i_d_head_tile * d_head_tile_size + ip_dkv, if_seq],
      dtype=out_dv_ref.dtype
    )

    nisa._tiled_offloaded_memcpy(
      dst=out_dk_ref[batch_id, head_id,
                     i_d_head_tile * d_head_tile_size + ip_dkv, if_seq],
      src=dk_acc_buf[i_d_head_tile * d_head_tile_size + ip_dkv, if_seq],
      dtype=out_dk_ref.dtype
    )

  # Write dQ
  ip_dq_ref, if_dq_ref = nl.mgrid[0:d_head, 0:seqlen]
  nisa._tiled_offloaded_memcpy(
    dst=out_dq_ref[batch_id, head_id, ip_dq_ref, if_dq_ref],
    src=recv_dq_buf[:, :],
    dtype=out_dq_ref.dtype
  )
  return out_dq_ref, out_dk_ref, out_dv_ref

@nki.jit
def ring_attention_fwd(q, k, v, seed,
                       rank_id=0,
                       src_tgt_pairs=[],
                       softmax_scale=None,
                       use_causal_mask=True,
                       mixed_precision=True,
                       dropout_p=0.0,
                       config: FlashConfig=FlashConfig()):
  """
  The NKI ring attention implementation on top of the flash attention.

  Inputs:
    q: query tensor of shape (b, h, d, seqlen_chunk)
    k: key tensor of shape (b, kv_heads, d, seqlen_chunk)
    v: value tensor of shape (b, kv_heads, d, seqlen_chunk)
    seed: seed tensor of shape (1,)

  Outputs:
    o: output buffer of shape (b, h, seqlen, d)
    lse: log-sum-exp for bwd pass stored in (b, h, nl.tile_size.pmax, seqlen // nl.tile_size.pmax) where nl.tile_size.pmax is 128

  Compile-time Constants:
    rank_id: The current worker rank, important when we use causal mask.
    src_tgt_pairs: The list describing the ring of communication represented by global rank IDs.
    Can be global pairs containing pairs that do not touch the current rank.
    Or rank pairs containing pairs that touch the current rank.
    softmax_scale: scaling for softmax, is None, default is `1.0/(d**0.5)`
    mixed_precision: flag to set non-matmul ops in fp32 precision, defualt is set to `true`, if false, we use same precision as input types
    causal_mask: flag to set causal masking
    config: dataclass with Performance config parameters for flash attention with default values
      seq_tile_size: `default=2048`, size of the kv tile size for attention computation
      reduction
      training: bool to indicate training vs inference `default=True`

  Performance Notes:
    For better performance, the kernel is tiled to be of size `LARGE_TILE_SZ`, and Flash attention math techniques are applied in unit
    of `LARGE_TILE_SZ`. Seqlen that is not divisible by `LARGE_TILE_SZ` is not supported at the moment.
  GQA support Notes: the spmd kernel for launching kernel should be on kv_heads instead of nheads
    ```
      e.g.
      MHA: q: [b, h, d, s], k: [b, h, d, s] , v: [b, h, s, d]
        usage: flash_fwd[b, h](q, k, v,...)
      GQA: q: [b, h, d, s], k: [b, kv_h, d, s] , v: [b, kv_h, s, d]
        usage: flash_fwd[b, kv_h](q, k, v,...)
    ```
  """
  B_F_SIZE=512
  B_P_SIZE=128
  b , h, d, seqlen  = q.shape
  B_D_SIZE = d
  k_h = k.shape[1]
  v_shape = v.shape
  assert config.should_transpose_v, f" require to use set the should_transpose_v in the FlashConfig"
  assert tuple(v_shape) == (b, k_h, d, seqlen), f"V shape does not match layout requirements, expect: {(b, k_h, d, seqlen)} but got {v_shape}"
  assert tuple(k.shape) == (b, k_h, d, seqlen), f" k and v shape does not match the layout defined in the function, but got {k.shape}"
  assert d <= 128, f" we do not support head_dim > 128, got head dim {d}"
  assert use_causal_mask, f" use without causal mask is not tested yet. "
  kernel_dtype = nl.bfloat16 if mixed_precision else q.dtype
  acc_type =  np.dtype(np.float32) if mixed_precision else kernel_dtype
  
  o = nl.ndarray((b, h, seqlen, d), dtype=q.dtype, buffer=nl.shared_hbm)
  if config.training:
    if config.lse_dtype:
        lse_dtype = getattr(nl, config.lse_dtype)
    else:
        lse_dtype = acc_type
    lse = nl.ndarray(
        (b, h, nl.tile_size.pmax, seqlen // nl.tile_size.pmax),
        dtype=lse_dtype,
        buffer=nl.shared_hbm,
    )
  else:
      lse = None
  
  cc_configs = CollectivesConfig(src_tgt_pairs=src_tgt_pairs)
  rank_src_tgt_pairs = cc_configs.rank_src_tgt_pairs[rank_id]
  global_ring_order = cc_configs.rank_recv_replica_groups[rank_id]
  
  num_workers = max(len(rank_src_tgt_pairs),1)
  local_rank_id = global_rank_to_local_rank(global_ring_order, rank_id)
  local_ring_order = global_ring_to_local_ring(global_ring_order)

  n_tile_q = seqlen // B_P_SIZE # since q will be loaded on PE
  q_h_per_k_h = h // k_h
  softmax_scale = softmax_scale or (1.0 / (d ** 0.5))

  LARGE_TILE_SZ = config.seq_tile_size
  assert config.seq_tile_size >= 512, f" seq tile_size {config.seq_tile_size} cannot be less than 512"
  assert seqlen % LARGE_TILE_SZ == 0, f"seqlen is not divisible by {LARGE_TILE_SZ}"
  num_large_k_tile = seqlen // LARGE_TILE_SZ
  REDUCTION_TILE = config.seq_tile_size // 2

  # inference flag, check if lse is none
  inference = not(config.training)
  if inference:
    assert lse is None, "lse should be none for inference"
    assert seed is None, f"seed should be None for inference, but got {seed}"
    assert dropout_p == 0.0, f"dropout should be 0.0 for inference but got {dropout_p}"
  else:
    assert lse is not None, "lse should not be none for training"

  if dropout_p > 0.0 and not inference:
    seed_local = nl.ndarray((par_dim(1), 1), buffer=nl.sbuf, dtype=nl.int32)
    seed_local[0, 0] = nl.load(seed[0])
    # TODO: Remove this once the dropout supports scale prob
    dropout_p_tensor = nl.full((B_P_SIZE, 1), fill_value=dropout_p, dtype=np.float32)
  else:
    dropout_p_tensor = None
    seed_local = None

  batch_id = nl.program_id(axis=0)
  head_id = nl.program_id(axis=1)

  # Virtual global flash attention accumulators
  o_buffer = nl.full((q_h_per_k_h, n_tile_q, num_large_k_tile * num_workers, par_dim(B_P_SIZE), d), 0.0,
                    dtype=acc_type, buffer=nl.sbuf, name="o_buffer") # zeros does not work
  l_buffer = nl.full((q_h_per_k_h, n_tile_q, num_large_k_tile * num_workers, par_dim(B_P_SIZE), 1), 0.0,
                    dtype=acc_type, buffer=nl.sbuf, name="l_buffer")
  m_buffer = nl.full((q_h_per_k_h, n_tile_q, num_large_k_tile * num_workers, par_dim(B_P_SIZE), 1), 0.0,
                    dtype=acc_type, buffer=nl.sbuf, name="m_buffer")

  # Double buffers to hold the sharded KV values
  send_k_buf = nl.ndarray((par_dim(d), seqlen), dtype=k.dtype, buffer=nl.private_hbm, name="send_k_buf")
  recv_k_buf = nl.ndarray((par_dim(d), seqlen), dtype=k.dtype, buffer=nl.private_hbm, name="recv_k_buf")
  send_v_buf = nl.ndarray((par_dim(d), seqlen), dtype=v.dtype, buffer=nl.private_hbm, name="send_v_buf")
  recv_v_buf = nl.ndarray((par_dim(d), seqlen), dtype=v.dtype, buffer=nl.private_hbm, name="recv_v_buf")

  # kv_idx, kv_buf_ix, kv_buf_iy = nl.mgrid[0:2, 0:d, 0:n]
  # kv_idx = nl.arange(2)[None, :, None]
  kv_buf_ix = nl.arange(d)[:, None]
  kv_buf_iy = nl.arange(seqlen)[None, :]

  i_f_128 = nl.arange(B_P_SIZE)[None, :]
  i_f_d = nl.arange(B_D_SIZE)[None, :]
  i_p_d = nl.arange(B_D_SIZE)[:,None]
  i_q_p = nl.arange(B_P_SIZE)[:,None]
  i_0_f = nl.arange(1)[None, :]

  # Calculate the local self-attention from qkv inputs directly
  for i_q_h in nl.affine_range(q_h_per_k_h):
    cur_k_tile = nl.ndarray((par_dim(B_D_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
    cur_v_tile = nl.ndarray((LARGE_TILE_SZ // B_P_SIZE, par_dim(B_P_SIZE), B_D_SIZE), dtype=kernel_dtype)
    load_tile_size = B_P_SIZE
    # First tile
    for k_i in nl.affine_range(LARGE_TILE_SZ // load_tile_size):
      load_p, load_f = nl.mgrid[0:B_D_SIZE, 0:load_tile_size]
      cur_k_tile[load_p, load_tile_size * k_i + load_f] = nl.load(
        k[batch_id, head_id, load_p, load_tile_size * k_i + load_f]
      )
    for v_i in nl.affine_range(LARGE_TILE_SZ // load_tile_size):
      load_p, load_f = nl.mgrid[0:B_D_SIZE, 0:B_P_SIZE]
      store_p, store_f = nl.mgrid[0:B_P_SIZE, 0:B_D_SIZE]
      loaded = nl.load(v[batch_id, head_id, load_p, B_P_SIZE * v_i + load_f], dtype=kernel_dtype)
      cur_v_tile[v_i, store_p, store_f] = nisa.nc_transpose(loaded)

    for i in nl.affine_range(n_tile_q):
      q_tile = nl.ndarray((B_D_SIZE, B_P_SIZE),dtype=kernel_dtype)
      q_tile[i_p_d, i_f_128] = nl.load(q[batch_id, head_id * q_h_per_k_h + i_q_h, i_p_d, i * B_P_SIZE + i_f_128], dtype=kernel_dtype) \
                                * softmax_scale # load (d, 128) tile in SBUF
      _flash_attention_core(q_local_tile=q_tile, k=cur_k_tile, v=cur_v_tile,
                            q_h_per_k_h=q_h_per_k_h, seqlen_q=seqlen,
                            o_buffer=o_buffer[i_q_h, i], l_buffer=l_buffer[i_q_h, i], m_buffer=m_buffer[i_q_h, i],
                            batch_id=batch_id, head_id=head_id,
                            gqa_head_idx=i_q_h, q_tile_idx=local_rank_id * n_tile_q + i,
                            global_k_large_tile_idx=local_rank_id * num_large_k_tile,
                            local_k_large_tile_idx=0,
                            olm_buffer_idx=0,
                            kernel_dtype=kernel_dtype, acc_type=acc_type,
                            flash_config=config, use_causal_mask=use_causal_mask,
                            initialize=True,
                            B_P_SIZE=B_P_SIZE, B_F_SIZE=B_F_SIZE, B_D_SIZE=B_D_SIZE,
                            dropout_p=dropout_p, dropout_p_tensor=dropout_p_tensor, seed_tensor=seed_local)
    
    # Rest of the tiles
    for j in nl.sequential_range(1, num_large_k_tile):
      cur_k_tile = nl.ndarray((par_dim(B_D_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
      cur_v_tile = nl.ndarray((LARGE_TILE_SZ//B_P_SIZE, par_dim(B_P_SIZE), B_D_SIZE), dtype=kernel_dtype)
      load_tile_size = B_P_SIZE
      for k_i in nl.affine_range(LARGE_TILE_SZ // load_tile_size):
        load_p, load_f = nl.mgrid[0:B_D_SIZE, 0:load_tile_size]
        cur_k_tile[load_p, load_tile_size * k_i + load_f] = nl.load(
          k[batch_id, head_id, load_p, j * LARGE_TILE_SZ + load_tile_size * k_i + load_f]
        )
      for v_i in nl.affine_range(LARGE_TILE_SZ // load_tile_size):
        load_p, load_f = nl.mgrid[0:B_D_SIZE, 0:B_P_SIZE]
        store_p, store_f = nl.mgrid[0:B_P_SIZE, 0:B_D_SIZE]
        loaded = nl.load(v[batch_id, head_id, load_p, j * LARGE_TILE_SZ + B_P_SIZE * v_i + load_f], dtype=kernel_dtype)
        cur_v_tile[v_i, store_p, store_f] = nisa.nc_transpose(loaded)

      for i in nl.affine_range(n_tile_q):
        i_f_128 = nl.arange(B_P_SIZE)[None, :]
        i_f_d = nl.arange(B_D_SIZE)[None, :]
        i_p_d = nl.arange(B_D_SIZE)[:,None]
        q_tile = nl.ndarray((B_D_SIZE, B_P_SIZE),dtype=kernel_dtype)
        q_tile[i_p_d, i_f_128] = nl.load(q[batch_id, head_id * q_h_per_k_h + i_q_h, i_p_d, i * B_P_SIZE + i_f_128], dtype=kernel_dtype) \
                                  * softmax_scale # load (d, 128) tile in SBUF
        _flash_attention_core(q_local_tile=q_tile, k=cur_k_tile, v=cur_v_tile,
                              q_h_per_k_h=q_h_per_k_h, seqlen_q=seqlen,
                              o_buffer=o_buffer[i_q_h, i], l_buffer=l_buffer[i_q_h, i], m_buffer=m_buffer[i_q_h, i],
                              batch_id=batch_id, head_id=head_id,
                              gqa_head_idx=i_q_h, q_tile_idx=local_rank_id * n_tile_q + i,
                              global_k_large_tile_idx=local_rank_id * num_large_k_tile + j,
                              local_k_large_tile_idx=j,
                              olm_buffer_idx=j,
                              kernel_dtype=kernel_dtype, acc_type=acc_type,
                              flash_config=config, use_causal_mask=use_causal_mask,
                              initialize=False,
                              B_P_SIZE=B_P_SIZE, B_F_SIZE=B_F_SIZE, B_D_SIZE=B_D_SIZE,
                              dropout_p=dropout_p, dropout_p_tensor=dropout_p_tensor, seed_tensor=seed_local)

  # Prepare the send buffers
  nisa._tiled_offloaded_memcpy(dst=send_k_buf[kv_buf_ix, kv_buf_iy],
                               src=k[batch_id, head_id, kv_buf_ix, kv_buf_iy])
  nisa._tiled_offloaded_memcpy(dst=send_v_buf[kv_buf_ix, kv_buf_iy],
                               src=v[batch_id, head_id, kv_buf_ix, kv_buf_iy])
  # Keep receiving the K and V chunks from the left neighbor
  # TODO: Use sequential_range
  for ring_step in nl.static_range(1, num_workers):
    recv_local_rank = local_ring_order[ring_step]
    recv_global_rank = global_ring_order[ring_step]
    nccl.collective_permute(src=send_k_buf[kv_buf_ix, kv_buf_iy],
                            dst=recv_k_buf[kv_buf_ix, kv_buf_iy],
                            source_target_pairs=src_tgt_pairs)
    nccl.collective_permute(src=send_v_buf[kv_buf_ix, kv_buf_iy],
                            dst=recv_v_buf[kv_buf_ix, kv_buf_iy],
                            source_target_pairs=src_tgt_pairs)

    for i_q_h in nl.affine_range(q_h_per_k_h):
      for j in nl.sequential_range(num_large_k_tile):
        cur_k_tile = nl.ndarray((par_dim(B_D_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
        cur_v_tile = nl.ndarray((LARGE_TILE_SZ // B_P_SIZE, par_dim(B_P_SIZE), B_D_SIZE), dtype=kernel_dtype)
        load_tile_size = B_P_SIZE
        for k_i in nl.affine_range(LARGE_TILE_SZ // load_tile_size):
          load_p, load_f = nl.mgrid[0:B_D_SIZE, 0:load_tile_size]
          cur_k_tile[load_p, load_tile_size * k_i + load_f] = nl.load(
            recv_k_buf[load_p, j * LARGE_TILE_SZ + load_tile_size * k_i + load_f]
          )
        for v_i in nl.affine_range(LARGE_TILE_SZ // load_tile_size):
          load_p, load_f = nl.mgrid[0:B_D_SIZE, 0:B_P_SIZE]
          loaded = nl.load(recv_v_buf[load_p, j * LARGE_TILE_SZ + B_P_SIZE * v_i + load_f], dtype=kernel_dtype)
          store_p, store_f = nl.mgrid[0:B_P_SIZE, 0:B_D_SIZE]
          cur_v_tile[v_i, store_p, store_f] = nisa.nc_transpose(loaded)

        for i in nl.affine_range(n_tile_q):
          i_f_128 = nl.arange(B_P_SIZE)[None, :]
          i_f_d = nl.arange(B_D_SIZE)[None, :]
          i_p_d = nl.arange(B_D_SIZE)[:,None]
          q_tile = nl.ndarray((B_D_SIZE, B_P_SIZE),dtype=kernel_dtype)
          q_tile[i_p_d, i_f_128] = nl.load(q[batch_id, head_id * q_h_per_k_h + i_q_h, i_p_d, i * B_P_SIZE + i_f_128], dtype=kernel_dtype) \
                                    * softmax_scale # load (d, 128) tile in SBUF
          _flash_attention_core(q_local_tile=q_tile, k=cur_k_tile, v=cur_v_tile,
                                q_h_per_k_h=q_h_per_k_h, seqlen_q=seqlen,
                                o_buffer=o_buffer[i_q_h, i], l_buffer=l_buffer[i_q_h, i], m_buffer=m_buffer[i_q_h, i],
                                batch_id=batch_id, head_id=head_id,
                                gqa_head_idx=i_q_h, q_tile_idx=local_rank_id * n_tile_q + i,
                                global_k_large_tile_idx=recv_local_rank * num_large_k_tile + j,
                                local_k_large_tile_idx=j,
                                olm_buffer_idx=ring_step * num_large_k_tile + j,
                                kernel_dtype=kernel_dtype, acc_type=acc_type,
                                flash_config=config, use_causal_mask=use_causal_mask,
                                initialize=False,
                                B_P_SIZE=B_P_SIZE, B_F_SIZE=B_F_SIZE, B_D_SIZE=B_D_SIZE,
                                dropout_p=dropout_p, dropout_p_tensor=dropout_p_tensor, seed_tensor=seed_local)
    
    # Swap the buffer
    nisa._tiled_offloaded_memcpy(dst=send_k_buf[kv_buf_ix, kv_buf_iy],
                                 src=recv_k_buf[kv_buf_ix, kv_buf_iy])
    nisa._tiled_offloaded_memcpy(dst=send_v_buf[kv_buf_ix, kv_buf_iy],
                                 src=recv_v_buf[kv_buf_ix, kv_buf_iy])
  for i_q_h in nl.affine_range(q_h_per_k_h):
    for i in nl.affine_range(n_tile_q):
      # -------- write output to buffer on HBM ------------ #
      out = nl.ndarray((par_dim(B_P_SIZE), B_D_SIZE), dtype=kernel_dtype)
      out[i_q_p, i_f_d] = nl.multiply(o_buffer[i_q_h, i, num_workers * num_large_k_tile - 1, i_q_p, i_f_d],
                                      nl.exp(m_buffer[i_q_h, i, num_workers * num_large_k_tile - 1, i_q_p, i_0_f] - \
                                             l_buffer[i_q_h, i, num_workers * num_large_k_tile - 1, i_q_p, i_0_f]),
                                      dtype=kernel_dtype)

      nl.store(o[batch_id, head_id * q_h_per_k_h + i_q_h, i * B_P_SIZE + i_q_p, i_f_d], out[i_q_p, i_f_d])
      if not inference:
        lse_local = nl.zeros((par_dim(B_P_SIZE), 1), dtype=acc_type)
        lse_local[i_q_p, i_0_f] = nl.copy(l_buffer[i_q_h, i, num_workers * num_large_k_tile - 1, i_q_p, i_0_f], dtype=acc_type)
        nl.store(lse[batch_id, head_id * q_h_per_k_h + i_q_h, i_q_p, i + i_0_f], lse_local[i_q_p, i_0_f])
  if config.training:
    return o, lse
  else:
    return o

