"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

Kernels written by the AWS Neuron.

WARNING: These kernels:
   - Are tested only against internal nightly builds
   - May not be compatible with public NeuronSDK releases
   - Have not been extensively tested across all input configurations
   - Carry no compatibility guarantees
   - The behavior of these kernels may be modified without prior notice

Attention building blocks

"""
import numpy as np

from neuronxcc.nki import trace
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl

from neuronxcc.nki.language import par_dim
from dataclasses import dataclass

def div_ceil(n, d):
  return (n + d - 1) // d

@dataclass(frozen=True)
class FlashConfig:
  """
    Config class for flash attention with default values
  """
  seq_tile_size:int = 2048
  training:bool = True
  should_transpose_v:bool = False

  __annotations__ = {
    'seq_tile_size': int,
    'training': bool,
    'should_transpose_v': bool
  }

@trace
def _flash_attention_core(q_local_tile, k, v,
                          q_h_per_k_h, seqlen_q,
                          o_buffer, l_buffer, m_buffer,
                          batch_id, head_id, gqa_head_idx, q_tile_idx,
                          local_k_large_tile_idx,
                          kernel_dtype, acc_type,
                          flash_config: FlashConfig,
                          olm_buffer_idx=None,
                          global_k_large_tile_idx=None,
                          use_causal_mask=False, initialize=False,
                          B_P_SIZE=128, B_F_SIZE=512, B_D_SIZE=128,
                          dropout_p=0.0, dropout_p_tensor=None, seed_tensor=None
                          ):
  """
  The flash attention core function to calcualte self attention between a tile of q and a block of K and V.
  The q_local_tile has (B_P_SIZE, B_F_SIZE), which is loaded into the SBUF already. The block size of K and V
  is defined in the seq_tile_size of the flash_config. The results are stored in the following three buffers
  o_buffer: (num_large_k_tile, B_P_SIZE, d)
  l_buffer: (num_large_k_tile, B_P_SIZE, 1)
  m_buffer: (num_large_k_tile, B_P_SIZE, 1)
  """
  LARGE_TILE_SZ = flash_config.seq_tile_size
  REDUCTION_TILE = min(2048, LARGE_TILE_SZ // 2)
  num_k_tile_per_large_tile = LARGE_TILE_SZ // B_F_SIZE
  seqlen_k = k.shape[-1]
  seq_q_num_tiles = seqlen_q // B_P_SIZE
  seq_k_num_tiles = seqlen_k // B_F_SIZE

  # Indices used by the distributed attention
  if global_k_large_tile_idx is None:
    global_k_large_tile_idx = local_k_large_tile_idx
  if olm_buffer_idx is None:
    olm_buffer_idx = local_k_large_tile_idx

  i_q_p = nl.arange(B_P_SIZE)[:, None]
  i_q_f = nl.arange(B_F_SIZE)[None, :]
  i_d_p = nl.arange(B_D_SIZE)[:, None]
  i_d_f = nl.arange(B_D_SIZE)[None, :]
  i_f_128 = nl.arange(B_P_SIZE)[None, :]
  i_f_k_tiles = nl.arange(num_k_tile_per_large_tile)[None, :]

  # mask are used to only apply computation to the lower half of the matrix,
  # which reduce the arthimetic intensity by half
  forward_mask = q_tile_idx * B_P_SIZE >= global_k_large_tile_idx * LARGE_TILE_SZ if use_causal_mask else None
  # Negation mask is the negation of `forward_mask`, which is used for the
  # instructions executed on the blocks in the upper triangular section
  # of the matrix.
  # These instructions should not be executed when causual mask is disabled.
  #
  # For example, the o_buffer still needs to be propagated from o[j-1] to o[j] in
  # the upper triangular of the matrix.
  negation_mask = q_tile_idx * B_P_SIZE < global_k_large_tile_idx * LARGE_TILE_SZ if use_causal_mask else None

  qk_res_buf = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ), buffer=nl.sbuf, dtype=acc_type)
  max_local = nl.ndarray((par_dim(B_P_SIZE), num_k_tile_per_large_tile), dtype=acc_type)
  for k_i in nl.affine_range(num_k_tile_per_large_tile):
    qk_psum = nl.zeros((par_dim(B_P_SIZE), B_F_SIZE),
                        dtype=np.float32, buffer=nl.psum)  # (128, 512)
    multiplication_required_selection = global_k_large_tile_idx * LARGE_TILE_SZ + k_i * B_F_SIZE <= q_tile_idx * B_P_SIZE if use_causal_mask else None
    qk_psum[i_q_p, i_q_f] += nl.matmul(q_local_tile, k[i_d_p, k_i * B_F_SIZE + i_q_f], transpose_x=True,
                                       mask=multiplication_required_selection) # (p(128), 512)

    if use_causal_mask:
      left_diagonal_selection = q_tile_idx * B_P_SIZE >= global_k_large_tile_idx * LARGE_TILE_SZ + (k_i + 1) * B_F_SIZE
      diagonal_and_right_selection = (q_tile_idx * B_P_SIZE < global_k_large_tile_idx * LARGE_TILE_SZ + (k_i + 1) * B_F_SIZE) & forward_mask

      q_pos = q_tile_idx * B_P_SIZE + i_q_p
      k_pos = global_k_large_tile_idx * LARGE_TILE_SZ + k_i * B_F_SIZE + i_q_f
      pred = q_pos >= k_pos
      # For tiles on and to the right of the diagonal, need to do affine_select.
      # Magic number -9984.0 to replace -inf similar to what Tensorizer uses
      qk_res_buf[i_q_p, k_i * B_F_SIZE + i_q_f] = nisa.affine_select(
        pred=pred,
        on_true_tile=qk_psum[i_q_p, i_q_f], on_false_value=-9984.0, dtype=kernel_dtype,
        mask=diagonal_and_right_selection)

      # For tiles on the left of the diagonal, direct copy, no select required.
      qk_res_buf[i_q_p, k_i * B_F_SIZE + i_q_f] = \
        nl.copy(qk_psum[i_q_p, i_q_f], dtype=kernel_dtype, mask=left_diagonal_selection)
    else:
      # Simply send psum result back to sbuf
      qk_res_buf[i_q_p, k_i * B_F_SIZE + i_q_f] = \
        nl.copy(qk_psum[i_q_p, i_q_f], dtype=kernel_dtype)

    # Calculate max of the current tile
    max_local[i_q_p, k_i] = nisa.tensor_reduce(np.max, qk_res_buf[i_q_p, k_i * B_F_SIZE + i_q_f], axis=(1,),
                                        dtype=acc_type, negate=False, mask=forward_mask)

  max_ = nisa.tensor_reduce(np.max, max_local[i_q_p, i_f_k_tiles], axis=(1, ),
                    dtype=acc_type, negate=False, mask=forward_mask)
  if not initialize:
    m_previous = nl.copy(m_buffer[olm_buffer_idx - 1, i_q_p, 0])
    m_buffer[olm_buffer_idx, i_q_p, 0] = nl.maximum(m_previous, max_, mask=forward_mask) # (128,1)
    if use_causal_mask:
      m_buffer[olm_buffer_idx, i_q_p, 0] = nl.copy(m_previous, mask=negation_mask)

    m_current = m_buffer[olm_buffer_idx, i_q_p, 0]
    # Compute scaling factor
    alpha = nisa.activation(np.exp, m_previous, bias=-1*m_current, scale=1.0, mask=forward_mask)
    o_previous = nl.copy(o_buffer[olm_buffer_idx-1, i_q_p, i_d_f], mask=forward_mask)
    o_previous_scaled = nl.multiply(o_previous, alpha, mask=forward_mask)
  else:
    m_buffer[0, i_q_p, 0] = nl.copy(max_)
    m_current = max_

  p_local = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
  i_r_f = nl.arange(REDUCTION_TILE)[None,: ]
  p_partial_sum = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ // REDUCTION_TILE), dtype=acc_type)
  for k_r_i in nl.affine_range(LARGE_TILE_SZ // REDUCTION_TILE):
    # compute exp(qk-max)
    p_local[i_q_p, k_r_i * REDUCTION_TILE + i_r_f] = \
      nisa.activation(np.exp,
                      qk_res_buf[i_q_p, k_r_i * REDUCTION_TILE + i_r_f],
                      bias=-1 * m_current,
                      scale=1.0,
                      dtype=kernel_dtype,
                      mask=forward_mask)

    # dropout
    if dropout_p > 0.0:
      for k_d_i in nl.sequential_range(REDUCTION_TILE // B_F_SIZE):
        offset = k_d_i + k_r_i * (REDUCTION_TILE // B_F_SIZE) \
                  + global_k_large_tile_idx * (LARGE_TILE_SZ // B_F_SIZE) \
                  + q_tile_idx * seq_k_num_tiles \
                  + (head_id * q_h_per_k_h + gqa_head_idx) * seq_k_num_tiles * seq_q_num_tiles \
                  + batch_id * nl.num_programs(1) * seq_k_num_tiles * seq_q_num_tiles
        offset_seed = nl.add(seed_tensor[0, 0], offset, mask=forward_mask)
        nl.random_seed(seed=offset_seed, mask=forward_mask)
        softmax_dropout = nl.dropout(p_local[i_q_p, k_r_i * REDUCTION_TILE + k_d_i * B_F_SIZE + i_q_f],
                                    rate=dropout_p_tensor[i_q_p, 0],
                                    mask=forward_mask)
        p_local[i_q_p, k_r_i * REDUCTION_TILE + k_d_i * B_F_SIZE + i_q_f] = \
          nl.multiply(softmax_dropout, 1 / (1 - dropout_p), mask=forward_mask)

    # Compute partial row-tile sum of exp(qk-max))
    p_partial_sum[i_q_p, k_r_i] = nl.sum(p_local[i_q_p, k_r_i * REDUCTION_TILE + i_r_f], axis=1, dtype=acc_type, mask=forward_mask)

  p_local_transposed = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
  for i_p_t in nl.affine_range(LARGE_TILE_SZ // 512):
    p_local_t_tmp = nl.ndarray((par_dim(B_P_SIZE), 512), buffer=nl.psum, dtype=np.float32)
    for i_p_t_local in nl.affine_range(512//128):
      p_local_t_tmp[i_q_p, i_p_t_local*128 + i_f_128] = nisa.nc_transpose(p_local[i_q_p, i_p_t*512+i_p_t_local * B_P_SIZE + i_f_128], mask=forward_mask)
    i_f_512 = nl.arange(512)[None, :]
    p_local_transposed[i_q_p, i_p_t * 512 + i_f_512 ] = nl.copy(p_local_t_tmp[i_q_p, i_f_512], dtype=kernel_dtype, mask=forward_mask)

  ps = nl.sum(p_partial_sum, axis=1, dtype=acc_type, mask=forward_mask)
  pv_psum = nl.zeros((par_dim(B_P_SIZE), B_D_SIZE), dtype=np.float32, buffer=nl.psum)
  for k_i in nl.affine_range(LARGE_TILE_SZ // B_P_SIZE):
    pv_psum[i_q_p, i_d_f] += nl.matmul(p_local_transposed[i_q_p, k_i * B_P_SIZE + i_f_128],
                                       v[k_i, i_q_p, i_d_f],
                                       transpose_x=True,
                                       mask=forward_mask) # (128, 128) (p(Br), d)

  if initialize:
    o_buffer[olm_buffer_idx, i_q_p, i_d_f] = nl.copy(pv_psum[i_q_p, i_d_f])
    l_buffer[olm_buffer_idx, i_q_p, 0] = nl.add(nl.log(ps), max_)
  else:
    if use_causal_mask:
      o_buffer[olm_buffer_idx, i_q_p, i_d_f] = nl.copy(o_buffer[olm_buffer_idx-1, i_q_p, i_d_f], mask=negation_mask)
    o_buffer[olm_buffer_idx, i_q_p, i_d_f] = nl.add(o_previous_scaled, pv_psum, mask=forward_mask)

    l_prev = l_buffer[olm_buffer_idx-1, i_q_p, 0]
    l_exp = nl.add(nl.exp(nl.subtract(l_prev, m_current, mask=forward_mask), mask=forward_mask), ps, mask=forward_mask)
    l_buffer[olm_buffer_idx, i_q_p, 0] = nl.add(m_current, nl.log(l_exp, mask=forward_mask), mask=forward_mask)
    if use_causal_mask:
      l_buffer[olm_buffer_idx, i_q_p, 0] = nl.copy(l_buffer[olm_buffer_idx-1, i_q_p, 0], mask=negation_mask)


@trace
def _flash_attn_bwd_core(
  q_local, k_local, transposed_k_local, v_local, dy_local,
  dk_psum, dv_psum, dq_local_reduced,
  softmax_exp_bias, dy_o_sum,
  local_i_q_seq_tile, local_i_k_seq_tile,
  seqlen_q, seqlen_k, d_head,
  use_causal_mask,
  kernel_dtype, mixed_dtype,
  softmax_scale,
  seed_local, dropout_p, dropout_p_local,
  global_i_q_seq_tile = None,
  global_i_k_seq_tile = None,
  # Used for nl.loop_reduce on dQ if local_i_k_seq_tile is not an index e.g. if it has an offset
  local_i_k_seq_tile_for_dq_reduce = None,
):
  """
  The flash backward core function to calculate the gradients of Q, K and V
  of the given tiles. The result will be accumulated into the dk, dv, dq psum
  """
  q_seq_n_tiles, q_seq_tile_size = div_ceil(seqlen_q, 128), 128
  d_head_n_tiles, d_head_tile_size = div_ceil(d_head, 128), min(d_head, 128)
  if seqlen_k >= 512:
    k_seq_n_tiles, k_seq_tile_size = seqlen_k // 512, 512
  else:
    k_seq_n_tiles, k_seq_tile_size = seqlen_k // 128, 128
  k_seq_n_tiles_backward, k_seq_tile_size_backward = seqlen_k // 128, 128
  k_seq_fwd_bwd_tile_multipler = k_seq_tile_size // k_seq_tile_size_backward

  if global_i_q_seq_tile is None:
    global_i_q_seq_tile = local_i_q_seq_tile
    global_i_k_seq_tile = local_i_k_seq_tile
  
  if local_i_k_seq_tile_for_dq_reduce is None:
    local_i_k_seq_tile_for_dq_reduce = local_i_k_seq_tile

  mask = global_i_q_seq_tile * q_seq_tile_size >= global_i_k_seq_tile * k_seq_tile_size if use_causal_mask else None
  # PSUM buffer shape: [q_seq_tile_size P, k_seq_tile_size F]
  qk_psum = nl.zeros((par_dim(q_seq_tile_size), k_seq_tile_size),
                      dtype=np.float32, buffer=nl.psum)
  qk_res_buf = nl.ndarray((par_dim(q_seq_tile_size), k_seq_tile_size), buffer=nl.sbuf, dtype=kernel_dtype)

  batch_id = nl.program_id(axis=0)
  head_id = nl.program_id(axis=1)
  # Tensor indices for accessing qk result in k_seq_tile_size
  if_q = nl.arange(q_seq_tile_size)[None, :]
  ip_qk = nl.arange(d_head_tile_size)[:, None]

  ip_q = nl.arange(q_seq_tile_size)[:, None]
  if_k = nl.arange(k_seq_tile_size)[None, :]

  # Loop over contraction dim of QK matmul
  for i_d_head_tile in nl.affine_range(d_head_n_tiles):
    ##############################################################
    # Step 2.1 Compute Q^T@K, with matmul(stationary=tensor_q, moving=tensor_k, contract=d_head)
    ##############################################################
    qk_psum[ip_q, if_k] += nisa.nc_matmul(q_local[i_d_head_tile, ip_qk, if_q],
                                            k_local[i_d_head_tile, ip_qk, if_k],
                                            mask=mask)

  ######################################
  # Step 2.2. Apply optional causal mask
  ######################################
  if use_causal_mask:
    # Magic number -9984.0 to replace -inf similar to what Tensorizer uses
    qk_res_buf[ip_q, if_k] = nisa.affine_select(
      pred=(global_i_q_seq_tile * q_seq_tile_size + ip_q >= global_i_k_seq_tile * k_seq_tile_size + if_k),
      on_true_tile=qk_psum[ip_q, if_k], on_false_value=-9984.0, dtype=mixed_dtype,
      mask=mask)
  else:
    # Simply send psum result back to sbuf
    qk_res_buf[ip_q, if_k] = \
      nl.copy(qk_psum[ip_q, if_k], dtype=mixed_dtype)

  softmax_y = nl.ndarray((par_dim(q_seq_tile_size), k_seq_tile_size), dtype=kernel_dtype, buffer=nl.sbuf)
  softmax_y[ip_q, if_k] = nisa.activation(np.exp,
                                            data=qk_res_buf[ip_q, if_k],
                                            bias=softmax_exp_bias[local_i_q_seq_tile, ip_q, 0],
                                            scale=1.0,
                                            mask=mask)
  #####################################################################
  # Dropout
  #####################################################################
  if dropout_p > 0.0:
    offset = global_i_k_seq_tile + global_i_q_seq_tile * k_seq_n_tiles \
              + head_id * k_seq_n_tiles * q_seq_n_tiles \
              + batch_id * nl.num_programs(1) * k_seq_n_tiles * q_seq_n_tiles
    offset_seed = nl.add(seed_local[0, 0], offset, mask=mask)
    nl.random_seed(seed=offset_seed, mask=mask)
    softmax_y[ip_q, if_k] = nl.dropout(softmax_y[ip_q, if_k], rate=dropout_p_local[ip_q, 0], mask=mask)
    softmax_y[ip_q, if_k] = nl.multiply(softmax_y[ip_q, if_k], 1 / (1 - dropout_p), mask=mask)

  #####################################################################
  # Step 3.1 Calculate the backward gradients dL/dV, where y=softmax@V
  # in value projection with matmul(stationary=dy, moving=softmax)
  #####################################################################
  for i_d_head_tile in nl.affine_range(d_head_n_tiles):
    ip_dv = nl.arange(d_head_tile_size)[:, None]
    if_dv = nl.arange(k_seq_tile_size)[None, :]
    if_trans_dy = nl.arange(q_seq_tile_size)[None, :]
    trans_dy = nisa.nc_transpose(dy_local[i_d_head_tile, ip_dv, if_trans_dy],
                                  mask=mask)
    dv_psum[i_d_head_tile, ip_dv, if_dv] += \
      nisa.nc_matmul(trans_dy, softmax_y[ip_q, if_k], mask=mask)

  #####################################################################
  # Step 3.2 Calculate the backward gradients dL/dsoftmax, where y=softmax@V
  # in value projection with matmul(stationary=dy, moving=v)
  #####################################################################
  softmax_dy_psum = nl.zeros((par_dim(q_seq_tile_size), k_seq_tile_size),
                              dtype=np.float32, buffer=nl.psum)
  for i_d_head_tile in nl.affine_range(d_head_n_tiles):
    ip_softmax_dy = nl.arange(d_head_tile_size)[:, None]
    if_dy = nl.arange(q_seq_tile_size)[None, :]
    softmax_dy_psum[ip_q, if_k] += \
      nisa.nc_matmul(dy_local[i_d_head_tile, ip_softmax_dy, if_dy],
                      v_local[i_d_head_tile, ip_softmax_dy, if_k],
                      mask=mask)

  softmax_dy = nl.ndarray((par_dim(q_seq_tile_size), k_seq_tile_size), dtype=kernel_dtype, buffer=nl.sbuf)
  softmax_dy[ip_q, if_k] = nl.copy(softmax_dy_psum[ip_q, if_k], dtype=kernel_dtype,
                                      mask=mask)

  #####################################################################
  # Step 4 Calculate the softmax backward gradients dL/dx, where y=softmax(x)
  # dL/dx = y * (dL/dy - rowsum(dO_O)), where y = softmax(x)
  #####################################################################
  softmax_dx_local = nl.ndarray((par_dim(q_seq_tile_size), k_seq_tile_size), dtype=kernel_dtype, buffer=nl.sbuf)
  softmax_dx_local[ip_q, if_k] = \
    nisa.scalar_tensor_tensor(data=softmax_dy[ip_q, if_k],
                              op0=np.subtract,
                              operand0=dy_o_sum[local_i_q_seq_tile, ip_q, 0],
                              op1=np.multiply,
                              operand1=softmax_y[ip_q, if_k],
                              mask=mask)

  #####################################################################
  # Step 5.1 Calculate dK, with matmul(stationary=Q, moving=softmax_dx)
  #####################################################################
  for i_d_head_tile in nl.affine_range(d_head_n_tiles):
    ip_trans_q = nl.arange(d_head_tile_size)[:, None]
    if_trans_q = nl.arange(q_seq_tile_size)[None, :]
    ip_dk = nl.arange(d_head_tile_size)[:, None]
    trans_q_local = nisa.nc_transpose(q_local[i_d_head_tile, ip_trans_q, if_trans_q],
                                      mask=mask)
    dk_psum[i_d_head_tile, ip_dk, if_k] += \
      nisa.nc_matmul(trans_q_local,
                      softmax_dx_local[ip_q, if_k],
                      mask=mask)

  #####################################################################
  # Step 5.2 Calculate dQ
  #####################################################################
  if_k = nl.arange(k_seq_tile_size_backward)[None, :]
  ip_dq = nl.arange(d_head_tile_size)[:, None]
  if_dq = nl.arange(q_seq_tile_size)[None, :]
  if_d = nl.arange(d_head_tile_size)[None, :]
  ip_transposed_k = nl.arange(k_seq_tile_size_backward)[:, None]
  for i_d_head_tile in nl.affine_range(d_head_n_tiles):
    dq_psum = nl.zeros((par_dim(d_head_tile_size), q_seq_tile_size),
                        dtype=np.float32, buffer=nl.psum)
    for i_k_seq_tile_backward in nl.affine_range(k_seq_fwd_bwd_tile_multipler):
      transposed_softmax_dx_local = \
        nisa.nc_transpose(softmax_dx_local[ip_q, i_k_seq_tile_backward * k_seq_tile_size_backward + if_k],
                          mask=mask)
      dq_psum[ip_dq, if_dq] += nisa.nc_matmul(
          transposed_k_local[i_k_seq_tile_backward, i_d_head_tile, ip_transposed_k, if_d],
          transposed_softmax_dx_local,
          mask=mask)
    dq_local = nl.multiply(dq_psum[ip_dq, if_dq], softmax_scale, dtype=kernel_dtype, mask=mask)
    dq_local_reduced[local_i_q_seq_tile, i_d_head_tile, ip_dq, if_dq] = nl.loop_reduce(
      dq_local, op=np.add, loop_indices=(local_i_k_seq_tile_for_dq_reduce,),
      dtype=mixed_dtype, mask=mask)
