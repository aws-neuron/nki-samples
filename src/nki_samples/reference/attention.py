"""
Copyright (c) 2023, Amazon.com. All Rights Reserved

kernels - Builtin high performance attention kernels

"""
import numpy as np

import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc import nki

from neuronxcc.nki.language import par_dim
from dataclasses import dataclass
from functools import reduce as functools_reduce
from operator import mul as operator_mul


def n_elts(shape):
  return functools_reduce(operator_mul, shape, 1)


def linearize(shape, indices):
  return sum(i * (n_elts(shape[dim + 1:]))
             for dim, i in enumerate(indices))


def div_ceil(n, d):
  return (n + d - 1) // d


@dataclass(frozen=True)
class FlashConfig:
  """
    Config class for flash attention with default values
  """
  seq_tile_size:int = 2048
  attn_core_tile_size:int = 256
  training:bool = True
  should_transpose_v:bool = False
  lse_dtype: str = ""


@nki.jit(mode='trace')
def transpose_p_local(p_local_transposed, p_local, LARGE_TILE_SZ, use_dma_transpose=False):
  for i in nl.affine_range(LARGE_TILE_SZ // 512):
    # Temporarily disable use_dma_tranpose by default until we stablized it
    if use_dma_transpose and nisa.get_nc_version() >= nisa.nc_version.gen3:
      p_local_t_tmp = nl.ndarray((par_dim(128), 512), buffer=nl.sbuf, dtype=p_local.dtype)
    else:
      p_local_t_tmp = nl.ndarray((par_dim(128), 512), buffer=nl.psum, dtype=np.float32)

    for j in nl.affine_range(512 // 128):
      j_128_slice = nl.ds(j * 128, 128)
      i_j_128_slice = nl.ds(i * 512 + j * 128, 128)

      if use_dma_transpose and nisa.get_nc_version() >= nisa.nc_version.gen3:
        p_local_t_tmp[:, j_128_slice] = nisa.dma_transpose(
          p_local[:, i_j_128_slice])
      else:
        p_local_t_tmp[:, j_128_slice] = nisa.nc_transpose(
          p_local[:, i_j_128_slice])

    p_local_transposed[:, nl.ds(i * 512, 512)] = nl.copy(
      p_local_t_tmp, dtype=p_local_transposed.dtype)


@nki.jit(mode='trace')
def dropout_p_local(p_local, dropout_p, dropout_p_tensor, seed_tensor,
                    seed_offset_base, k_r_i, REDUCTION_TILE):
  B_F_SIZE = 512
  for k_d_i in nl.sequential_range(REDUCTION_TILE // B_F_SIZE):
    p_local_f_slice = nl.ds(k_r_i * REDUCTION_TILE + k_d_i * B_F_SIZE, B_F_SIZE)

    offset = k_d_i + seed_offset_base
    offset_seed = nl.add(seed_tensor, offset, dtype=nl.int32)
    nl.random_seed(seed=offset_seed)
    softmax_dropout = nl.dropout(p_local[:, p_local_f_slice],
                                 rate=dropout_p_tensor[:, 0])
    p_local[:, p_local_f_slice] = nl.multiply(
      softmax_dropout, 1 / (1 - dropout_p))


@nki.jit(mode='trace')
def _flash_attention_core(q_local_tile, k, v,
                          q_h_per_k_h, seqlen_q, nheads,
                          o_buffer, l_buffer, m_buffer,
                          batch_id, head_id, gqa_head_idx, q_tile_idx,
                          local_k_large_tile_idx,
                          kernel_dtype, acc_type,
                          flash_config: FlashConfig,
                          use_causal_mask, sliding_window,
                          B_P_SIZE=128, B_F_SIZE=512, B_D_SIZE=128,
                          dropout_p=0.0, dropout_p_tensor=None, seed_tensor=None,
                          logit_bias_tile=None):
  """
  The flash attention core function to calcualte self attention between a tile of q and a block of K and V.
  The q_local_tile has (B_P_SIZE, B_F_SIZE), which is loaded into the SBUF already. The block size of K and V
  is defined in the seq_tile_size of the flash_config. The results are stored in the following three buffers
  o_buffer: (B_P_SIZE, d)
  l_buffer: (B_P_SIZE, 1)
  m_buffer: (B_P_SIZE, 1)
  """
  NEG_INFINITY = nl.fp32.min
  LARGE_TILE_SZ = flash_config.seq_tile_size
  num_k_tile_per_large_tile = LARGE_TILE_SZ // B_F_SIZE
  seqlen_k = k.shape[-1]
  seq_q_num_tiles = seqlen_q // B_P_SIZE
  seq_k_num_tiles = seqlen_k // B_F_SIZE

  qk_res_buf = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ), buffer=nl.sbuf, dtype=acc_type)
  max_local = nl.ndarray((par_dim(B_P_SIZE), num_k_tile_per_large_tile), dtype=acc_type)

  for k_i in nl.affine_range(num_k_tile_per_large_tile):
    k_i_b_f_slice = nl.ds(k_i * B_F_SIZE, B_F_SIZE)

    qk_psum = nl.ndarray((par_dim(B_P_SIZE), B_F_SIZE),
                        dtype=np.float32, buffer=nl.psum)  # (128, 512)
    if use_causal_mask:
      multiplication_required_selection = q_tile_idx * B_P_SIZE >= local_k_large_tile_idx * LARGE_TILE_SZ + k_i * B_F_SIZE
    else:
      multiplication_required_selection = True

    if multiplication_required_selection:
      qk_psum[:, :] = nl.matmul(q_local_tile, k[:, k_i_b_f_slice], transpose_x=True) # (p(128), 512)
    else:
      qk_psum[:, :] = 0

    if use_causal_mask:
      left_diagonal_selection = q_tile_idx * B_P_SIZE >= local_k_large_tile_idx * LARGE_TILE_SZ + (k_i + 1) * B_F_SIZE
      diagonal_and_right_selection = (q_tile_idx * B_P_SIZE < local_k_large_tile_idx * LARGE_TILE_SZ + (k_i + 1) * B_F_SIZE)
      right_diagonal_selection = ((q_tile_idx + 1) * B_P_SIZE <= local_k_large_tile_idx * LARGE_TILE_SZ + k_i * B_F_SIZE)
      diagonal_and_left_selection = ((q_tile_idx + 1) * B_P_SIZE > local_k_large_tile_idx * LARGE_TILE_SZ + k_i * B_F_SIZE)
      diagonal = ((q_tile_idx * B_P_SIZE < local_k_large_tile_idx * LARGE_TILE_SZ + (k_i + 1) * B_F_SIZE) &
                  ((q_tile_idx + 1) * B_P_SIZE > local_k_large_tile_idx * LARGE_TILE_SZ + k_i * B_F_SIZE))

      i_q_p, i_q_f = nl.mgrid[0:B_P_SIZE, 0:B_F_SIZE]
      q_pos = q_tile_idx * B_P_SIZE + i_q_p
      k_pos = local_k_large_tile_idx * LARGE_TILE_SZ + k_i * B_F_SIZE + i_q_f
      pred_causal = q_pos >= k_pos  # causal mask
      pred_sliding = k_pos > q_pos - sliding_window  # sliding window mask

      qk_select_tmp = nl.ndarray(qk_psum.shape, dtype=qk_psum.dtype, buffer=nl.sbuf)

      if logit_bias_tile is not None:
        if right_diagonal_selection:
          qk_select_tmp[...] = qk_psum

          # For tiles to the right of the diagonal, do affine_select.
          qk_res_buf[:, k_i_b_f_slice] = nisa.affine_select(
              pred=pred_causal,
              on_true_tile=qk_select_tmp, on_false_value=NEG_INFINITY, dtype=acc_type)

        # For tiles on the diagonal, add logit bias and need to do affine_select.
        intermediate = \
            nl.add(qk_psum, logit_bias_tile[:, k_i_b_f_slice],
                   dtype=acc_type, mask=diagonal)
        qk_res_buf[:, k_i_b_f_slice] = nisa.affine_select(
            pred=pred_causal,
            on_true_tile=intermediate, on_false_value=NEG_INFINITY, dtype=acc_type,
            mask=diagonal)

        # For tiles on the left of the diagonal, add logit bias.
        qk_res_buf[:, k_i_b_f_slice] = \
            nl.add(qk_psum, logit_bias_tile[:, k_i_b_f_slice],
                   dtype=acc_type, mask=left_diagonal_selection)

        if sliding_window > 0:  # Apply sliding window mask
          qk_res_buf[:, k_i_b_f_slice] = nisa.affine_select(
              pred=pred_sliding,
              on_true_tile=intermediate, on_false_value=NEG_INFINITY, dtype=acc_type,
              mask=left_diagonal_selection) 
      else:
        # Apply causal mask
        qk_res_buf[:, k_i_b_f_slice] = nisa.affine_select(pred=pred_causal, 
                                                          on_true_tile=qk_psum, 
                                                          on_false_value=NEG_INFINITY,
                                                          dtype=acc_type)
        if sliding_window > 0:  # Apply sliding window mask
          qk_res_buf[:, k_i_b_f_slice] = nisa.affine_select(pred=pred_sliding, 
                                                            on_true_tile=qk_res_buf[:, k_i_b_f_slice], 
                                                            on_false_value=NEG_INFINITY, 
                                                            dtype=acc_type,
                                                            mask=diagonal_and_left_selection)
    else:
      if logit_bias_tile is not None:
        # Simply add logit bias which copies back to sbuf at the same time
        qk_res_buf[:, k_i_b_f_slice] = \
            nl.add(qk_psum, logit_bias_tile[:, k_i_b_f_slice], dtype=acc_type)
      else:
        # Simply send psum result back to sbuf
        qk_res_buf[:, k_i_b_f_slice] = nl.copy(qk_psum, dtype=acc_type)

    # Calculate max of the current tile
    max_local[:, k_i] = nisa.tensor_reduce(
      np.max, qk_res_buf[:, k_i_b_f_slice], axis=(1,), dtype=acc_type,
      negate=False)

  max_ = nisa.tensor_reduce(np.max, max_local[:, :], axis=(1, ),
                            dtype=acc_type, negate=False)

  o_previous_scaled = nl.ndarray((par_dim(B_P_SIZE), B_D_SIZE), dtype=o_buffer.dtype)

  m_previous = nl.copy(m_buffer[:, 0])
  m_buffer[:, 0] = nl.maximum(m_previous, max_) # (128,1)

  m_current = m_buffer[:, 0]
  # Compute scaling factor
  alpha = nisa.activation(np.exp, m_current, bias=m_previous, scale=-1.0)
  o_previous_scaled[...] = nl.multiply(o_buffer[:, :], alpha)

  p_local = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
  REDUCTION_TILE = min(2048, LARGE_TILE_SZ // 2)

  p_partial_sum = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ // REDUCTION_TILE), dtype=acc_type)

  for k_r_i in nl.affine_range(LARGE_TILE_SZ // REDUCTION_TILE):
    k_r_i_reduce_slice = nl.ds(k_r_i * REDUCTION_TILE, REDUCTION_TILE)

    # dropout
    if dropout_p > 0.0:
      # compute exp(qk-max)
      p_local[:, k_r_i_reduce_slice] = \
        nisa.activation(np.exp, qk_res_buf[:, k_r_i_reduce_slice],
                        bias=-1 * m_current, scale=1.0,
                        dtype=kernel_dtype)

      seed_offset_base = k_r_i * (REDUCTION_TILE // B_F_SIZE) \
                         + local_k_large_tile_idx * (LARGE_TILE_SZ // B_F_SIZE) \
                         + q_tile_idx * seq_k_num_tiles \
                         + (head_id * q_h_per_k_h + gqa_head_idx) * seq_k_num_tiles * seq_q_num_tiles \
                         + batch_id * nheads * seq_k_num_tiles * seq_q_num_tiles

      dropout_p_local(p_local=p_local, dropout_p=dropout_p,
                      dropout_p_tensor=dropout_p_tensor, seed_tensor=seed_tensor,
                      seed_offset_base=seed_offset_base, k_r_i=k_r_i,
                      REDUCTION_TILE=REDUCTION_TILE)

      # Compute partial row-tile sum of exp(qk-max))
      # FIXME: Use activation accumulate and accumulate over k_r_i loop?
      p_partial_sum[:, k_r_i] = nl.sum(p_local[:, k_r_i_reduce_slice],
                                       axis=1, dtype=acc_type)
    else:
      # compute exp(qk-max)
      # Compute partial row-tile sum of exp(qk-max))
      # FIXME: Use activation accumulate to accumulate over k_r_i loop?
      p_local[:, k_r_i_reduce_slice] = \
        nisa.activation_reduce(np.exp, qk_res_buf[:, k_r_i_reduce_slice],
                               bias=-1 * m_current, scale=1.0,
                               reduce_op=nl.add, reduce_res=p_partial_sum[:, k_r_i],
                               dtype=kernel_dtype)

  ps = nl.sum(p_partial_sum, axis=1, dtype=acc_type)

  p_local_transposed = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
  transpose_p_local(p_local_transposed=p_local_transposed, p_local=p_local,
                    LARGE_TILE_SZ=LARGE_TILE_SZ)

  pv_psum = nl.zeros((par_dim(B_P_SIZE), B_D_SIZE), dtype=np.float32,
                     buffer=nl.psum, lazy_initialization=True)
  for k_i in nl.affine_range(LARGE_TILE_SZ // B_P_SIZE):
    pv_psum[:, :] += nl.matmul(p_local_transposed[:, nl.ds(k_i * B_P_SIZE, B_P_SIZE)],
                               v[k_i, :, :], transpose_x=True) # (128, 128) (p(Br), d)

  o_buffer[:, :] = nl.add(o_previous_scaled, pv_psum)

  exp = nisa.activation(nl.exp, m_current, bias=l_buffer[:, 0], scale=-1.0)
  l_buffer[:, 0] = nl.add(m_current, nisa.activation(nl.log, exp, bias=ps))


@nki.jit(mode='trace')
def load_v_tile(v_hbm_tile, cur_v_tile, j, v_i, config):
  LARGE_TILE_SZ = config.seq_tile_size
  B_P_SIZE = 128

  if not config.should_transpose_v:
    cur_v_tile[v_i, :, :] = nl.load(
      v_hbm_tile[nl.ds(j * LARGE_TILE_SZ + B_P_SIZE * v_i, B_P_SIZE), :],
      dtype=cur_v_tile.dtype)
    return

  if nisa.get_nc_version() >= nisa.nc_version.gen3:
    cur_v_tile_transposed = nisa.dma_transpose(
      v_hbm_tile[:, nl.ds(j * LARGE_TILE_SZ + B_P_SIZE * v_i, B_P_SIZE)])
    cur_v_tile[v_i, :, :] = nisa.tensor_copy(cur_v_tile_transposed,
                                             dtype=cur_v_tile.dtype)
    return

  cur_v_tile[v_i, :, :] = nl.load_transpose2d(
    v_hbm_tile[:, nl.ds(j * LARGE_TILE_SZ + B_P_SIZE * v_i, B_P_SIZE)],
    dtype=cur_v_tile.dtype)



@nki.jit
def flash_fwd(q, k, v, seed, logit_bias=None,
              softmax_scale=None,
              use_causal_mask=True,
              sliding_window=-1,
              mixed_precision=True,
              dropout_p=0.0, config=None):
  """
  Flash Attention Forward kernel

  IO tensor layouts:
    - q: shape   (bs, n_heads, d, seq_q)
    - k: shape   (bs, nk_heads, d, seq_k)
    - v: shape   (bs, nv_heads, d, seq_v) if config.should_transpose_v  else (bs, nv_heads, seq_v, d)
    - seed: shape (1,)
    - logit_bias: shape (bs, n_heads, seq_q, seq_k)
    - o: shape (bs, n_heads, seq_q, d)
    - lse: shape (bs, n_heads, nl.tile_size.pmax, seq // nl.tile_size.pmax) if training else None
    - This kernel requires seq_k == seq_v

  IO tensor dtypes:
    - This kernel assumes all IO tensors have the same dtype
    - If mixed_precision is True, then all Tensor Engine operation will be performed in
      bfloat16 and accumulation will be performed in float32. Otherwise the intermediates
      will be in the same type as the inputs.

  Compile-time Constants:
    - softmax_scale: scaling for softmax, is None, default is `1.0/(d**0.5)`
    - mixed_precision: flag to set non-matmul ops in fp32 precision, default is set to `true`, if false, we use same precision as input types
    - use_causal_mask: flag to set causal masking
    - sliding_window: causal (or left) sliding window size, default is -1, which means sliding window is off.
        when turned on (sliding_window > 0), only the last previous `sliding_window` tokens are attended to. See more in Masking support Notes below.
    - config: Instance of :class:`nki.kernels.attention.FlashConfig` with Performance config parameters for flash attention with default values
        seq_tile_size: `default=2048`, size of the kv tile size for attention computation reduction
        training: bool to indicate training vs inference `default=True`

  Performance Notes:
    For better performance, the kernel is tiled to be of size `config.seq_tile_size`, and Flash attention math techniques are applied in unit
    of `config.seq_tile_size`. Seqlen that is not divisible by `config.seq_tile_size` is not supported at the moment.

    For large seqlen, `o_buffer` will overflow the statebuf. the kernel is tile `o_buffer` based on the value of `config.attn_core_tile_size`.
    This is a tradeoff between memory usage and performance. The default value of `config.attn_core_tile_size` is 256, which means the `o_buffer`
    will roughly take half of the statebuf. The computes are also tiled accordingly. DMA will be rematerialized
    `seqlen_q // B_P_SIZE // attn_core_tile_size times`.



  GQA support Notes:
    the spmd kernel for launching kernel should be on kv_heads instead of nheads
  
  Masking support Notes:
    3 masking options are supported:
      1. use_causal_mask=False, sliding_window=-1: full (no masking)
      2. use_causal_mask=True, sliding_window=-1: causal
      3. use_causal_mask={True/False}, sliding_window > 0: causal & sliding window 
          - including current token, attend only the previous `sliding_window` tokens

          e.g. seq_q = seq_k = 5, sliding_window = 2, the attn mask applied on QK^T is:

          [[1 0 0 0 0]    # token 0 attends to [0]
           [1 1 0 0 0]    # token 1 attends to [0,1]
           [0 1 1 0 0]    # token 2 attends to [1,2]
           [0 0 1 1 0]    # token 3 attends to [2,3]
           [0 0 0 1 1]]   # token 4 attends to [3,4]

          - given sliding_window > 0, use_causal_mask is overriden to be True 
              i.e. no support for bidirectional sliding window

  Example usage:
    MHA: q: [b, h, d, s], k: [b, h, d, s], v: [b, h, s, d]
      usage: `flash_fwd[b, h](q, k, v, ...)`
    GQA: q: [b, h, d, s], k: [b, kv_h, d, s], v: [b, kv_h, s, d]
      usage: `flash_fwd[b, kv_h](q, k, v, ...)`
  """
  config = config or FlashConfig()
  B_F_SIZE=512
  B_P_SIZE=128
  b, h, d, seqlen_q  = q.shape
  B_D_SIZE = d
  _, k_h, _, seqlen_k = k.shape
  if config.should_transpose_v:
    assert tuple(v.shape) == (b, k_h, d, seqlen_k), f"Expect shape of V to be {(b, k_h, d, seqlen_k)} (batch, heads, d_head, seqlen_k) but got {v.shape}"
    assert tuple(k.shape) == (b, k_h, d, seqlen_k), f"Expect shape of K to be {(b, k_h, d, seqlen_k)} (batch, heads, d_head, seqlen_k) but got {k.shape}"
  else:
    assert tuple(v.shape) == (b, k_h, seqlen_k, d), f"Expect shape of V to be {(b, k_h, seqlen_k, d)} (batch, heads, seqlen_k, d_head) but got {v.shape}"
    assert tuple(k.shape) == (b, k_h, d, seqlen_k), f"Expect shape of K to be {(b, k_h, d, seqlen_k)} (batch, heads, d_head, seqlen_k) but got {k.shape}"
  assert d <= 128, f" we do not support head_dim > 128, got head dim {d}"
  use_causal_mask = True if sliding_window > 0 else use_causal_mask  # setting sliding window assumes causal
  kernel_dtype = nl.bfloat16 if mixed_precision else q.dtype
  acc_type = np.dtype(np.float32) if mixed_precision else kernel_dtype

  o = nl.ndarray((b, h, seqlen_q, d), dtype=q.dtype, buffer=nl.shared_hbm)
  if config.training:
    if config.lse_dtype:
      lse_dtype = getattr(nl, config.lse_dtype)
    else:
      lse_dtype = acc_type
    lse = nl.ndarray((b, h, nl.tile_size.pmax, seqlen_q // nl.tile_size.pmax),
                     dtype=lse_dtype, buffer=nl.shared_hbm)
  else:
    lse = None

  assert nl.program_ndim() == 2,\
    f'Expect spmd grid with 2 dimensions, got {nl.program_ndim()} instead!'
  batch_id = nl.program_id(axis=0)
  head_id = nl.program_id(axis=1)

  softmax_scale = softmax_scale or (1.0 / (d ** 0.5))

  n_tile_q = seqlen_q // B_P_SIZE # since q will be loaded on tensor engine

  LARGE_TILE_SZ = config.seq_tile_size
  attn_core_tile_size = config.attn_core_tile_size

  # FIXME: Add masking for different seqlen values.
  assert config.seq_tile_size >= 512, f" seq tile_size {config.seq_tile_size} cannot be less than 512"
  assert seqlen_k % LARGE_TILE_SZ == 0, f"Need seqlen_k to be divisible by {LARGE_TILE_SZ} but got {seqlen_k}"
  num_large_k_tile = seqlen_k // LARGE_TILE_SZ

  # inference flag, check if lse is none
  inference = not config.training
  if inference:
    assert lse is None, "lse should be none for inference"
    assert seed is None, f"seed should be None for inference, but got {seed}"
    assert dropout_p==0.0, f"dropout should be 0.0 for inference but got {dropout_p}"
  else:
    assert lse is not None, "lse should not be none for training"
  q_h_per_k_h = h // k_h

  if dropout_p > 0.0 and not inference:
    seed_local = nl.load(seed[0])
    # TODO: Remove this once the dropout supports scale prob
    dropout_p_tensor = nl.full((B_P_SIZE, 1), fill_value=dropout_p, dtype=np.float32)
  else:
    dropout_p_tensor = None
    seed_local = None

  if logit_bias is not None:
    b_logit_bias, h_logit_bias, _, _ = logit_bias.shape
    assert b_logit_bias == 1 and h_logit_bias == 1, "only support broadcasting logit_bias with batch 1, n_heads 1"

  n_remat = div_ceil(n_tile_q, attn_core_tile_size)
  attn_core_tile_size = min(n_tile_q, attn_core_tile_size)

  for i_q_h in nl.affine_range(q_h_per_k_h):
    # =============== Global Flash Attention accumulators ====================== #
    l_buffer = nl.full((par_dim(B_P_SIZE), n_tile_q), fill_value=nl.fp32.min, dtype=acc_type,
                        buffer=nl.sbuf, lazy_initialization=False)
    # =============== Global Flash Attention accumulators END ================== #

    for i0 in nl.sequential_range(n_remat):
      # =============== Global Flash Attention accumulators ====================== #
      o_buffer = nl.zeros((attn_core_tile_size, par_dim(B_P_SIZE), d), dtype=acc_type,
                          buffer=nl.sbuf, lazy_initialization=False)
      m_buffer = nl.full((attn_core_tile_size, par_dim(B_P_SIZE), 1), fill_value=nl.fp32.min,
                          dtype=acc_type,
                          buffer=nl.sbuf, lazy_initialization=False)
      # =============== Global Flash Attention accumulators END ================== #

      for j in nl.sequential_range(0, num_large_k_tile):
        cur_k_tile = nl.ndarray((par_dim(B_D_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
        cur_v_tile = nl.ndarray((LARGE_TILE_SZ // B_P_SIZE, par_dim(B_P_SIZE), B_D_SIZE), dtype=kernel_dtype)

        cur_k_tile[:, :] = nl.load(k[batch_id, head_id, :, nl.ds(j*LARGE_TILE_SZ, LARGE_TILE_SZ)])

        load_tile_size = B_P_SIZE

        v_hbm_tile = v[batch_id, head_id]
        for v_i in nl.affine_range(LARGE_TILE_SZ // load_tile_size):
          load_v_tile(v_hbm_tile=v_hbm_tile, cur_v_tile=cur_v_tile, j=j, v_i=v_i,
                      config=config)

        for i1 in nl.affine_range(attn_core_tile_size):
          i = i0 * attn_core_tile_size + i1
          # mask are used to only apply computation to the lower half of the matrix,
          # which reduce the arthimetic intensity by half.
          # forward_mask imply initialize, i.e. if forward_mask is false, initialize will
          # be false as well
          if use_causal_mask and sliding_window < 0:
            causal_mask = i * B_P_SIZE >= j * LARGE_TILE_SZ
            sliding_mask = True
          elif sliding_window > 0:
            causal_mask = i * B_P_SIZE >= j * LARGE_TILE_SZ
            sliding_mask = ((j+1) * LARGE_TILE_SZ - 1) > ((i * B_P_SIZE) - sliding_window)
          else:
            causal_mask = True
            sliding_mask = True
          
          if (i < n_tile_q) & causal_mask & sliding_mask:
            q_tile = nl.ndarray((B_D_SIZE, B_P_SIZE),dtype=kernel_dtype)
            q_hbm_tile = q[batch_id, head_id * q_h_per_k_h + i_q_h]
            q_sbuf_tile = nl.load(q_hbm_tile[:, nl.ds(i * B_P_SIZE, B_P_SIZE)],
                                  dtype=kernel_dtype) # load (d, 128) tile in SBUF
            q_tile[:, :] = q_sbuf_tile * softmax_scale

            logit_bias_tile = None
            if logit_bias is not None:
              logit_bias_tile = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
              logit_bias_tile[:, :] = nl.load(
                logit_bias[0, 0, nl.ds(i * B_P_SIZE, B_P_SIZE),
                           nl.ds(j * LARGE_TILE_SZ, LARGE_TILE_SZ)])

            _flash_attention_core(q_local_tile=q_tile, k=cur_k_tile, v=cur_v_tile,
                                  q_h_per_k_h=q_h_per_k_h, seqlen_q=seqlen_q, nheads=h,
                                  o_buffer=o_buffer[i1], l_buffer=l_buffer[:, i], m_buffer=m_buffer[i1],
                                  batch_id=batch_id, head_id=head_id,
                                  gqa_head_idx=i_q_h, q_tile_idx=i, local_k_large_tile_idx=j,
                                  kernel_dtype=kernel_dtype, acc_type=acc_type,
                                  flash_config=config, 
                                  use_causal_mask=use_causal_mask, sliding_window=sliding_window,
                                  B_P_SIZE=B_P_SIZE, B_F_SIZE=B_F_SIZE, B_D_SIZE=B_D_SIZE,
                                  dropout_p=dropout_p, dropout_p_tensor=dropout_p_tensor,
                                  seed_tensor=seed_local, logit_bias_tile=logit_bias_tile)

      # -------- write output to buffer on HBM ------------ #
      for i1 in nl.affine_range(attn_core_tile_size):
        i = i0 * attn_core_tile_size + i1

        if i < n_tile_q:
          exp = nisa.activation(np.exp, l_buffer[:, i], bias=m_buffer[i1, :, :],
                                scale=-1.0)
          out = nl.multiply(o_buffer[i1, :, :], exp,
                            dtype=kernel_dtype)

          nl.store(o[batch_id, head_id * q_h_per_k_h + i_q_h,
                     nl.ds(i*B_P_SIZE, B_P_SIZE), :], out)

    if not inference:
      nl.store(lse[batch_id, head_id * q_h_per_k_h + i_q_h, :, :], l_buffer[:, :])

  if config.training:
    return o, lse

  return o



@nki.jit
def flash_attn_bwd(
  q_ref, k_ref, v_ref, o_ref,
  dy_ref,
  lse_ref,
  seed_ref,
  logit_bias_ref=None,
  use_causal_mask=False,
  mixed_precision=False,
  dropout_p=0.0,
  softmax_scale=None,
):
  """
  Flash attention backward kernel. Compute the backward gradients.

  IO tensor layouts:
   - q_ref: shape (bs, nheads, head_size, seq)
   - k_ref: shape (bs, nheads, head_size, seq)
   - v_ref: shape (bs, nheads, head_size, seq)
   - o_ref: shape (bs, nheads, head_size, seq)
   - dy_ref: shape (bs, nheads, head_size, seq)
   - lse_ref: shape (bs, nheads, nl.tile_size.pmax, seq // nl.tile_size.pmax)
   - seed_ref: shape (1,)
   - logit_bias_ref: shape (bs, n_heads, seq_q, seq_k)
   - out_dq_ref: shape (bs, nheads, head_size, seq)
   - out_dk_ref: shape (bs, nheads, head_size, seq)
   - out_dv_ref: shape (bs, nheads, head_size, seq)

  Detailed steps:
    1. D = rowsum(dO ◦ O) (pointwise multiply)

    2. Recompute (softmax(Q^T@K + logic_bias))

      2.1 Q^T@K
      2.2 Scale the QK score
      2.3 Apply causal mask and add logit_bias
      2.4 softmax

    3. Compute the gradients of y = score @ V with respect to the loss

    4. Compute the gradients of y = softmax(x)

    5. Compute the gradients of Q^T@K

      4.1 Compute dQ
      4.2 Compute dK
  """

  # Use q_ref dtype as the intermediate tensor dtype
  # Assume all IO tensors have the same dtype
  kernel_dtype = q_ref.dtype
  mixed_dtype = np.dtype(np.float32) if mixed_precision else kernel_dtype

  assert q_ref.dtype == k_ref.dtype == v_ref.dtype == o_ref.dtype == dy_ref.dtype

  # Shape checking
  bs, nheads, d_head, seqlen_q = q_ref.shape
  _, _, _, seqlen_k = k_ref.shape
  assert tuple(k_ref.shape) == (bs, nheads, d_head, seqlen_k), \
    f"Input K shape mismatch, got {k_ref.shape}"
  assert tuple(v_ref.shape) == (bs, nheads, d_head, seqlen_k), \
    f"Input V shape mismatch, got {v_ref.shape}"
  assert tuple(o_ref.shape) == (bs, nheads, d_head, seqlen_q), \
    f"Input o shape mismatch, got {o_ref.shape}"
  assert tuple(dy_ref.shape) == (bs, nheads, d_head, seqlen_q), \
    f"Input dy shape mismatch, got {dy_ref.shape}"
  assert tuple(lse_ref.shape) == (bs, nheads, nl.tile_size.pmax, seqlen_q // nl.tile_size.pmax), \
    f"Input lse shape mismatch, got {lse_ref.shape}"
  if seed_ref is not None:
    assert tuple(seed_ref.shape) == (1,), \
      f"Input seed shape mismatch, got {seed_ref.shape}"

  out_dq_ref = nl.ndarray((bs, nheads, d_head, seqlen_q), dtype=q_ref.dtype,
                          buffer=nl.shared_hbm)
  out_dk_ref = nl.ndarray((bs, nheads, d_head, seqlen_k), dtype=q_ref.dtype,
                          buffer=nl.shared_hbm)
  out_dv_ref = nl.ndarray((bs, nheads, d_head, seqlen_k), dtype=q_ref.dtype,
                          buffer=nl.shared_hbm)

  # FIXME: Add masking for different seqlen values.
  assert seqlen_q % 128 == 0 and seqlen_k % 128 == 0, \
    f"Input sequence lengths must be divisible by 128, got seqlen_q == {seqlen_q} and seqlen_k == {seqlen_k}"

  # Softmax scaling factor, multiplied onto Q
  softmax_scale = softmax_scale or 1.0 / float(d_head ** 0.5)

  assert nl.program_ndim() == 2,\
    f'Expect spmd grid with 2 dimensions, got {nl.program_ndim()} instead!'
  # Different batch samples/attention heads have independent attention
  batch_id = nl.program_id(axis=0)
  head_id = nl.program_id(axis=1)

  assert nl.num_programs(1) == nheads, \
    f"The grid shape mismatch, got {nl.num_programs(1)} but should be {nheads}"

  if logit_bias_ref is not None:
    b_logit_bias, h_logit_bias, _, _ = logit_bias_ref.shape
    assert b_logit_bias == 1 and h_logit_bias == 1, "Only support broadcasting logit_bias with batch 1, n_heads 1"

  q_seq_n_tiles, q_seq_tile_size = div_ceil(seqlen_q, 128), 128
  d_head_n_tiles, d_head_tile_size = div_ceil(d_head, 128), min(d_head, 128)

  if seqlen_k >= 512:
    k_seq_n_tiles, k_seq_tile_size = seqlen_k // 512, 512
  else:
    k_seq_n_tiles, k_seq_tile_size = seqlen_k // 128, 128

  k_seq_n_tiles_backward, k_seq_tile_size_backward = seqlen_k // 128, 128
  k_seq_fwd_bwd_tile_multipler = k_seq_tile_size // k_seq_tile_size_backward

  ##############################################################
  # Step 2.4 Prefetch exp bias for softmax
  ##############################################################
  softmax_exp_bias = nl.zeros((par_dim(q_seq_tile_size), q_seq_n_tiles), dtype=mixed_dtype)
  lse_local = nl.load(lse_ref[batch_id, head_id, :, :], dtype=mixed_dtype)
  softmax_exp_bias[:, :] = lse_local * -1.0

  ##############################################################
  # Step 1 Compute rowsum(dO ◦ O)
  ##############################################################
  dy_o_sum = nl.ndarray((q_seq_n_tiles, par_dim(q_seq_tile_size), 1), dtype=mixed_dtype)
  compute_rowsum(dy_o_sum=dy_o_sum,
                 dy_ref_hbm_tile=dy_ref[batch_id, head_id],
                 o_ref_hbm_tile=o_ref[batch_id, head_id],
                 d_head_n_tiles=d_head_n_tiles, d_head_tile_size=d_head_tile_size,
                 q_seq_n_tiles=q_seq_n_tiles, q_seq_tile_size=q_seq_tile_size)

  if dropout_p > 0.0:
    seed_local = nl.load(seed_ref[0])
    # TODO: Remove this once the dropout supports scale prob
    dropout_p_local = nl.full((q_seq_tile_size, 1), fill_value=dropout_p, dtype=np.float32)
  else:
    seed_local = None
    dropout_p_local = None

  dq_local_reduced = nl.zeros((q_seq_n_tiles, d_head_n_tiles, par_dim(d_head_tile_size), q_seq_tile_size),
                              dtype=mixed_dtype)

  # affine_range give the compiler permission to vectorize instructions
  # inside the loop which improves the performance. However, when using the
  # the dropout we should use sequential_range to avoid setting
  # seed vectorization. TODO: the compiler should avoid vectorizing seed setting
  _range = nl.sequential_range if dropout_p > 0.0 else nl.affine_range

  for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
    i_k_seq_dslice = nl.ds(i_k_seq_tile * k_seq_tile_size, k_seq_tile_size)

    # Prefetch V, K
    v_local = nl.zeros((d_head_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size),
                       dtype=kernel_dtype)
    k_local = nl.zeros((d_head_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size),
                       dtype=kernel_dtype)
    transposed_k_local = nl.zeros((k_seq_fwd_bwd_tile_multipler, d_head_n_tiles,
                                   par_dim(k_seq_tile_size_backward), d_head_tile_size),
                                  dtype=kernel_dtype)

    load_kv(k_ref_hbm_tile=k_ref[batch_id, head_id],
            v_ref_hbm_tile=v_ref[batch_id, head_id],
            k_local=k_local, transposed_k_local=transposed_k_local, v_local=v_local,
            d_head_n_tiles=d_head_n_tiles, d_head_tile_size=d_head_tile_size,
            i_k_seq_tile=i_k_seq_tile, k_seq_tile_size=k_seq_tile_size,
            k_seq_tile_size_backward=k_seq_tile_size_backward)

    # FIXME: Pass sbuf instead, we will have psum spilling in the current implementation
    dv_psum = nl.zeros((d_head_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size),
                        dtype=np.float32, buffer=nl.psum)
    dk_psum = nl.zeros((d_head_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size),
                        dtype=np.float32, buffer=nl.psum)
    for i_q_seq_tile in _range(q_seq_n_tiles):
      # Prefetch dy, Q
      dy_local = nl.zeros((d_head_n_tiles, par_dim(d_head_tile_size), q_seq_tile_size), dtype=kernel_dtype)
      q_local = nl.zeros((d_head_n_tiles, par_dim(d_head_tile_size), q_seq_tile_size), dtype=kernel_dtype)

      load_dy_q(dy_ref_hbm_tile = dy_ref[batch_id, head_id],
                q_ref_hbm_tile = q_ref[batch_id, head_id],
                dy_local=dy_local, q_local=q_local, d_head_n_tiles=d_head_n_tiles,
                d_head_tile_size=d_head_tile_size, i_q_seq_tile=i_q_seq_tile,
                q_seq_tile_size=q_seq_tile_size, softmax_scale=softmax_scale)

      logit_bias_tile = None
      if logit_bias_ref is not None:
        i_q_seq_dslice = nl.ds(i_q_seq_tile * q_seq_tile_size, q_seq_tile_size)
        logit_bias_tile = nl.ndarray((par_dim(q_seq_tile_size), k_seq_tile_size),
                                     buffer=nl.sbuf, dtype=kernel_dtype)
        logit_bias_tile[:, :] = nl.load(
          logit_bias_ref[0, 0, i_q_seq_dslice, i_k_seq_dslice])

      _flash_attn_bwd_core(
        q_local=q_local, k_local=k_local, transposed_k_local=transposed_k_local,
        v_local=v_local, dy_local=dy_local,
        dk_psum=dk_psum, dv_psum=dv_psum, dq_local_reduced=dq_local_reduced,
        softmax_exp_bias=softmax_exp_bias, dy_o_sum=dy_o_sum,
        local_i_q_seq_tile=i_q_seq_tile, local_i_k_seq_tile=i_k_seq_tile,
        seqlen_q=seqlen_q, seqlen_k=seqlen_k, d_head=d_head, nheads=nheads,
        use_causal_mask=use_causal_mask,
        kernel_dtype=kernel_dtype, mixed_dtype=mixed_dtype,
        softmax_scale=softmax_scale,
        seed_local=seed_local, dropout_p=dropout_p, dropout_p_local=dropout_p_local,
        logit_bias_tile=logit_bias_tile
      )

    # Write dK, dV
    store_dk_dv(out_dk_ref_hbm_tile=out_dk_ref[batch_id, head_id],
                out_dv_ref_hbm_tile=out_dv_ref[batch_id, head_id],
                local_dk=dk_psum, local_dv=dv_psum, i_k_seq_dslice=i_k_seq_dslice,
                d_head_n_tiles=d_head_n_tiles, d_head_tile_size=d_head_tile_size)

  # Write dQ
  for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
    for i_d_head_tile in nl.affine_range(d_head_n_tiles):
      i_q_seq_dslice = nl.ds(i_q_seq_tile * q_seq_tile_size, q_seq_tile_size)
      i_d_head_dslice = nl.ds(i_d_head_tile * d_head_tile_size, d_head_tile_size)
      nl.store(
        out_dq_ref[batch_id, head_id, i_d_head_dslice, i_q_seq_dslice],
        value=dq_local_reduced[i_q_seq_tile, i_d_head_tile, :, :],
      )

  return out_dq_ref, out_dk_ref, out_dv_ref


@nki.jit(mode='trace')
def load_dy_q(dy_ref_hbm_tile, q_ref_hbm_tile, dy_local, q_local, d_head_n_tiles, d_head_tile_size, i_q_seq_tile,
              q_seq_tile_size, softmax_scale):
  for i_d_head_tile in nl.affine_range(d_head_n_tiles):
    i_d_head_dslice = nl.ds(i_d_head_tile * d_head_tile_size, d_head_tile_size)
    i_q_seq_dslice = nl.ds(i_q_seq_tile * q_seq_tile_size, q_seq_tile_size)

    dy_local[i_d_head_tile, :, :] = nl.load(
      dy_ref_hbm_tile[i_d_head_dslice, i_q_seq_dslice],
      dtype=dy_local.dtype)

    q_local[i_d_head_tile, :, :] = nl.load(
      q_ref_hbm_tile[i_d_head_dslice, i_q_seq_dslice],
      dtype=q_local.dtype) * softmax_scale


@nki.jit(mode='trace')
def store_dk_dv(out_dk_ref_hbm_tile, out_dv_ref_hbm_tile, local_dk, local_dv,
                d_head_n_tiles, d_head_tile_size, i_k_seq_dslice):
  for i in nl.affine_range(d_head_n_tiles):
    i_d_head_dslice = nl.ds(i * d_head_tile_size, d_head_tile_size)

    nl.store(out_dv_ref_hbm_tile[i_d_head_dslice, i_k_seq_dslice],
             value=local_dv[i, :, :])

    nl.store(out_dk_ref_hbm_tile[i_d_head_dslice, i_k_seq_dslice],
             value=local_dk[i, :, :])


@nki.jit(mode='trace')
def load_kv(k_ref_hbm_tile, v_ref_hbm_tile, k_local, transposed_k_local, v_local,
            d_head_n_tiles, d_head_tile_size, i_k_seq_tile, k_seq_tile_size,
            k_seq_tile_size_backward):
  k_seq_fwd_bwd_tile_multipler = k_seq_tile_size // k_seq_tile_size_backward

  for i in nl.affine_range(d_head_n_tiles):
    i_d_head_dslice = nl.ds(i * d_head_tile_size, d_head_tile_size)
    i_k_seq_dslice = nl.ds(i_k_seq_tile * k_seq_tile_size, k_seq_tile_size)
    k_local[i, :, :] = nl.load(k_ref_hbm_tile[i_d_head_dslice, i_k_seq_dslice],
                                           dtype=k_local.dtype)
    v_local[i, :, :] = nl.load(v_ref_hbm_tile[i_d_head_dslice, i_k_seq_dslice],
                                           dtype=v_local.dtype)
    ##############################################################
    # Prefetch k transpose for the backward too
    ##############################################################
    for j in nl.affine_range(k_seq_fwd_bwd_tile_multipler):
      i_k_dslice = nl.ds(j * k_seq_tile_size_backward, k_seq_tile_size_backward)
      transposed_k_local[j, i, :, :] = nisa.nc_transpose(k_local[i, :, i_k_dslice])


@nki.jit(mode='trace')
def compute_rowsum(dy_o_sum, dy_ref_hbm_tile, o_ref_hbm_tile, d_head_n_tiles, d_head_tile_size, q_seq_n_tiles,
                   q_seq_tile_size):
  mixed_dtype = dy_o_sum.dtype
  for i in nl.affine_range(q_seq_n_tiles):
    dy_o_partial = nl.zeros((par_dim(q_seq_tile_size), d_head_n_tiles), dtype=mixed_dtype)
    for j in nl.affine_range(d_head_n_tiles):
      d_head_dslice = nl.ds(j * d_head_tile_size, d_head_tile_size)
      q_seq_dslice = nl.ds(i * q_seq_tile_size, q_seq_tile_size)

      dy_local = nl.load_transpose2d(dy_ref_hbm_tile[d_head_dslice, q_seq_dslice],
                                     dtype=mixed_dtype)
      o_local = nl.load_transpose2d(o_ref_hbm_tile[d_head_dslice, q_seq_dslice],
                                    dtype=mixed_dtype)

      dy_o = nl.multiply(dy_local, o_local, dtype=mixed_dtype)
      dy_o_partial[:, j] = nisa.tensor_reduce(np.add, data=dy_o, axis=(1,),
                                              dtype=mixed_dtype)

    dy_o_sum[i, :, 0] = nisa.tensor_reduce(
      np.add, data=dy_o_partial[:, :], axis=(1,), dtype=mixed_dtype)


@nki.jit(mode='trace')
def _flash_attn_bwd_core(
  q_local, k_local, transposed_k_local, v_local, dy_local,
  dk_psum, dv_psum, dq_local_reduced,
  softmax_exp_bias, dy_o_sum,
  local_i_q_seq_tile, local_i_k_seq_tile,
  seqlen_q, seqlen_k, d_head, nheads,
  use_causal_mask,
  kernel_dtype, mixed_dtype,
  softmax_scale,
  seed_local, dropout_p, dropout_p_local,
  logit_bias_tile=None):
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

  mask = local_i_q_seq_tile * q_seq_tile_size >= local_i_k_seq_tile * k_seq_tile_size if use_causal_mask else None
  # PSUM buffer shape: [q_seq_tile_size P, k_seq_tile_size F]
  qk_psum = nl.zeros((par_dim(q_seq_tile_size), k_seq_tile_size),
                      dtype=np.float32, buffer=nl.psum)
  qk_res_buf = nl.ndarray((par_dim(q_seq_tile_size), k_seq_tile_size), buffer=nl.sbuf, dtype=kernel_dtype)

  batch_id = nl.program_id(axis=0)
  head_id = nl.program_id(axis=1)

  # Loop over contraction dim of QK matmul
  for i_d_head_tile in nl.affine_range(d_head_n_tiles):
    ##############################################################
    # Step 2.1 Compute Q^T@K, with matmul(stationary=tensor_q, moving=tensor_k, contract=d_head)
    ##############################################################
    qk_psum[:, :] += nisa.nc_matmul(q_local[i_d_head_tile, :, :],
                                            k_local[i_d_head_tile, :, :],
                                            mask=mask)

  ######################################
  # Step 2.2. Apply optional causal mask
  ######################################
  if use_causal_mask:
    iq, ik = nl.mgrid[0:q_seq_tile_size, 0:k_seq_tile_size]
    causal_pred = (local_i_q_seq_tile * q_seq_tile_size + iq >= local_i_k_seq_tile * k_seq_tile_size + ik)
    if logit_bias_tile is not None:
      # Magic number -9984.0 to replace -inf similar to what Tensorizer uses
      intermediate = \
        nl.add(qk_psum[:, :], logit_bias_tile[:, :], dtype=mixed_dtype, mask=mask)
      qk_res_buf[:, :] = nisa.affine_select(
        pred=causal_pred, 
        on_true_tile=intermediate, on_false_value=-9984.0, dtype=mixed_dtype,
        mask=mask
      )

    else:
      # Magic number -9984.0 to replace -inf similar to what Tensorizer uses
      qk_res_buf[:, :] = nisa.affine_select(
        pred=causal_pred,
        on_true_tile=qk_psum[:, :], on_false_value=-9984.0, dtype=mixed_dtype,
        mask=mask)
  else:
    if logit_bias_tile is not None:
      # Simply add logit bias which copies back to sbuf at the same time
      qk_res_buf[:, :] = \
        nl.add(qk_psum[:, :], logit_bias_tile[:, :], dtype=mixed_dtype)
    else:
      # Simply send psum result back to sbuf
      qk_res_buf[:, :] = \
        nl.copy(qk_psum[:, :], dtype=mixed_dtype)

  softmax_y = nl.ndarray((par_dim(q_seq_tile_size), k_seq_tile_size), dtype=kernel_dtype, buffer=nl.sbuf)
  softmax_y[:, :] = nisa.activation(np.exp,
                                    data=qk_res_buf[:, :],
                                    bias=softmax_exp_bias[:, local_i_q_seq_tile],
                                    scale=1.0,
                                    mask=mask)
  #####################################################################
  # Dropout
  #####################################################################
  if dropout_p > 0.0:
    offset = local_i_k_seq_tile + local_i_q_seq_tile * k_seq_n_tiles \
              + head_id * k_seq_n_tiles * q_seq_n_tiles \
              + batch_id * nheads * k_seq_n_tiles * q_seq_n_tiles
    offset_seed = nl.add(seed_local[0, 0], offset, mask=mask)
    nl.random_seed(seed=offset_seed, mask=mask)
    softmax_y[:, :] = nl.dropout(softmax_y[:, :], rate=dropout_p_local[:, 0], mask=mask)
    softmax_y[:, :] = nl.multiply(softmax_y[:, :], 1 / (1 - dropout_p), mask=mask)

  #####################################################################
  # Step 3.1 Calculate the backward gradients dL/dV, where y=softmax@V
  # in value projection with matmul(stationary=dy, moving=softmax)
  #####################################################################
  for i_d_head_tile in nl.affine_range(d_head_n_tiles):
    trans_dy = nisa.nc_transpose(dy_local[i_d_head_tile, :, :],
                                  mask=mask)
    dv_psum[i_d_head_tile, :, :] += \
      nisa.nc_matmul(trans_dy, softmax_y[:, :], mask=mask)

  #####################################################################
  # Step 3.2 Calculate the backward gradients dL/dsoftmax, where y=softmax@V
  # in value projection with matmul(stationary=dy, moving=v)
  #####################################################################
  softmax_dy_psum = nl.zeros((par_dim(q_seq_tile_size), k_seq_tile_size),
                              dtype=np.float32, buffer=nl.psum)
  for i_d_head_tile in nl.affine_range(d_head_n_tiles):
    softmax_dy_psum[:, :] += \
      nisa.nc_matmul(dy_local[i_d_head_tile, :, :],
                      v_local[i_d_head_tile, :, :],
                      mask=mask)

  softmax_dy = nl.ndarray((par_dim(q_seq_tile_size), k_seq_tile_size), dtype=kernel_dtype, buffer=nl.sbuf)
  softmax_dy[:, :] = nl.copy(softmax_dy_psum[:, :], dtype=kernel_dtype,
                                      mask=mask)

  #####################################################################
  # Step 4 Calculate the softmax backward gradients dL/dx, where y=softmax(x)
  # dL/dx = y * (dL/dy - rowsum(dO_O)), where y = softmax(x)
  #####################################################################
  softmax_dx_local = nl.ndarray((par_dim(q_seq_tile_size), k_seq_tile_size), dtype=kernel_dtype, buffer=nl.sbuf)
  softmax_dx_local[:, :] = \
    nisa.scalar_tensor_tensor(data=softmax_dy[:, :],
                              op0=np.subtract,
                              operand0=dy_o_sum[local_i_q_seq_tile, :, 0],
                              op1=np.multiply,
                              operand1=softmax_y[:, :],
                              mask=mask)

  #####################################################################
  # Step 5.1 Calculate dK, with matmul(stationary=Q, moving=softmax_dx)
  #####################################################################
  for i_d_head_tile in nl.affine_range(d_head_n_tiles):
    trans_q_local = nisa.nc_transpose(q_local[i_d_head_tile, :, :],
                                      mask=mask)
    dk_psum[i_d_head_tile, :, :] += \
      nisa.nc_matmul(trans_q_local,
                      softmax_dx_local[:, :],
                      mask=mask)

  #####################################################################
  # Step 5.2 Calculate dQ
  #####################################################################
  for i_d_head_tile in nl.affine_range(d_head_n_tiles):
    dq_psum = nl.zeros((par_dim(d_head_tile_size), q_seq_tile_size),
                        dtype=np.float32, buffer=nl.psum)
    for i_k_seq_tile_backward in nl.affine_range(k_seq_fwd_bwd_tile_multipler):
      i_k_seq_dslice = nl.ds(i_k_seq_tile_backward * k_seq_tile_size_backward,
                             k_seq_tile_size_backward)
      transposed_softmax_dx_local = \
        nisa.nc_transpose(softmax_dx_local[:, i_k_seq_dslice],
                          mask=mask)
      dq_psum[:, :] += nisa.nc_matmul(
          transposed_k_local[i_k_seq_tile_backward, i_d_head_tile, :, :],
          transposed_softmax_dx_local,
          mask=mask)
    dq_local = nl.multiply(dq_psum[:, :], softmax_scale, dtype=kernel_dtype, mask=mask)
    dq_local_reduced[local_i_q_seq_tile, i_d_head_tile, :, :] = nl.loop_reduce(
      dq_local, op=np.add, loop_indices=(local_i_k_seq_tile,),
      dtype=mixed_dtype, mask=mask)


@nki.jit
def fused_self_attn_for_SD_small_head_size(q_ref, k_ref, v_ref, use_causal_mask=False,
                                           mixed_precision=True):
  """
  Fused self attention kernel for small head size Stable Diffusion workload.

  Computes softmax(QK^T)V. Decoder model can optionally include a causal mask
  application. Does not include QKV projection, output projection, dropout,
  residual connection, etc.

  This kernel is designed to be used for Stable Diffusion models where the
  n_heads is smaller or equal to 128. Assertion is thrown if `n_heads` does
  not satisfy the requirement.

  IO tensor layouts:
   - q_ptr: shape   (bs, n_heads, seq_q)
   - k_ptr: shape   (bs, seq_k, n_heads)
   - v_ptr: shape   (bs, seq_v, n_heads)
   - out_ptr: shape (bs, seq_q, n_heads)
   - We use seq_q and seq_k just for clarity, this kernel requires seq_q == seq_k

  IO tensor dtypes:
   - This kernel assumes all IO tensors have the same dtype
   - If mixed_precision is True, then all Tensor Engine operation will be performed in
     bfloat16 and accumulation will be performed in float32. Otherwise the intermediates
     will be in the same type as the inputs.
  """
  # Use q_ref dtype as the intermediate tensor dtype
  # Assume all IO tensors have the same dtype
  kernel_dtype = q_ref.dtype
  pe_in_dt = nl.bfloat16 if mixed_precision else np.float32
  assert q_ref.dtype == k_ref.dtype == v_ref.dtype

  # Shape checking
  bs, d_head, seqlen = q_ref.shape
  assert d_head <= 128, "Cannot use this kernel for d_head > 128"
  assert tuple(q_ref.shape) == (bs, d_head, seqlen), 'Input shape mismatch!'
  assert tuple(k_ref.shape) == (bs, seqlen, d_head), 'Input shape mismatch!'
  assert tuple(v_ref.shape) == (bs, seqlen,  d_head), \
    f'Input shape mismatch! Expected: {(bs, seqlen, d_head)} Actual: {tuple(v_ref.shape)}'

  out_ref = nl.ndarray((bs, seqlen, d_head), dtype=q_ref.dtype, buffer=nl.shared_hbm)

  # Softmax scaling factor, multiplied onto Q
  softmax_scale = 0.125

  # Different batch samples/attention heads have independent attention
  batch_id = nl.program_id(axis=0)
  # batch_id = 0

  # TODO: make q_seq_tile_size user input
  # The matmuls currently use a fixed tile size of (128, 128). This may not achieve the best
  # performance for dense attention. However, since this kernel is in preparation
  # for block-sparse attention, this tile size is acceptable because the block
  # size of block-sparse attention cannot be too large.
  q_seq_n_tiles, q_seq_tile_size = seqlen // 128, 128
  k_seq_n_tiles, k_seq_tile_size = seqlen // 128, 128
  # No tiling on d_head dimension since the number of d_head fits in SB
  d_head_tile_size = d_head
  v_seq_n_tiles, v_seq_tile_size = seqlen // 128, 128

  ###################################
  # Step 1. transpose(tensor_v)
  ###################################
  # Buffer for v matrix transposed
  # Pre-fetch and keep it in SBUF throughout different softmax tiles
  trans_v = nl.ndarray((par_dim(v_seq_tile_size), v_seq_n_tiles, d_head), dtype=pe_in_dt)

  for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
    ip_v = nl.arange(v_seq_tile_size)[:, None]
    if_v = nl.arange(d_head_tile_size)[None, :]
    trans_v[ip_v, i_k_seq_tile, if_v] = nl.load(
      v_ref[batch_id, i_k_seq_tile * k_seq_tile_size + ip_v, if_v],
      dtype=pe_in_dt)

  q_local = nl.ndarray((q_seq_n_tiles, par_dim(d_head_tile_size), q_seq_tile_size), dtype=pe_in_dt)
  ip_q = nl.arange(d_head_tile_size)[:, None]
  if_q = nl.arange(q_seq_tile_size)[None, :]
  for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
    q_local[i_q_seq_tile, ip_q, if_q] = nl.load(
      q_ref[batch_id, ip_q, i_q_seq_tile * q_seq_tile_size + if_q],
      dtype=pe_in_dt) * softmax_scale

  k_local = nl.ndarray((k_seq_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size), dtype=pe_in_dt)
  ip_k = nl.arange(d_head_tile_size)[:, None]
  if_k = nl.arange(k_seq_tile_size)[None, :]
  for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
    k_local[i_k_seq_tile, ip_k, if_k] = nl.load_transpose2d(
      k_ref[batch_id,
            i_k_seq_tile * k_seq_tile_size + nl.arange(k_seq_tile_size)[:, None],
            nl.arange(d_head_tile_size)[None, :]],
      dtype=pe_in_dt)

  for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):  # indent = 2
    # A SBUF buffer for an independent softmax tile
    qk_res_buf = nl.ndarray((par_dim(q_seq_tile_size), seqlen), dtype=kernel_dtype)

    neg_max_res = nl.ndarray((par_dim(q_seq_tile_size), k_seq_n_tiles), dtype=kernel_dtype)
    ip_max = nl.arange(q_seq_tile_size)[:, None]
    if_max = nl.arange(k_seq_n_tiles)[None, :]

    # Loop over RHS free of matmul(stationary=tensor_q, moving=tensor_k, contract=d_head)
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):  # indent = 4

      # Since the K^T tile is the RHS, the q_seq_len dimension will be P in the result
      # PSUM buffer shape: [q_seq_tile_size P, k_seq_tile_size F]
      qk_psum = nl.zeros((par_dim(q_seq_tile_size), k_seq_tile_size),
                         dtype=np.float32, buffer=nl.psum)

      # Tensor indices for accessing qk result in k_seq_tile_size
      ip_qk = nl.arange(q_seq_tile_size)[:, None]
      if_qk = nl.arange(k_seq_tile_size)[None, :]

      ##############################################################
      # Step 2. matmul(stationary=tensor_q, moving=tensor_k, contract=d_head)
      ##############################################################
      qk_psum[ip_qk, if_qk] += nisa.nc_matmul(moving=k_local[i_k_seq_tile, ip_k, if_k],
                                              stationary=q_local[i_q_seq_tile, ip_q, if_q])

      ###################################
      # Step 3. Apply optional causal mask
      ###################################
      if use_causal_mask:
        # Magic number -9984.0 to replace -inf similar to what Tensorizer uses
        qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = nisa.affine_select(
          pred=(i_q_seq_tile * q_seq_tile_size + ip_qk >= i_k_seq_tile * k_seq_tile_size + if_qk),
          on_true_tile=qk_psum[ip_qk, if_qk], on_false_value=-9984.0, dtype=kernel_dtype)
      else:
        # Simply send psum result back to sbuf
        qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = nl.copy(qk_psum[ip_qk, if_qk],
                                                                              dtype=kernel_dtype)

      ###################################
      # Step 4. Softmax
      ###################################
      # TODO: use TensorScalarCacheReduce to avoid an extra copy
      # We want to break this reduction in tiles because we want to overlap it with the previous matmul
      neg_max_res[ip_max, i_k_seq_tile] = nisa.tensor_reduce(
        np.max, data=qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk],
        axis=(1,), dtype=kernel_dtype, negate=True)

    neg_max_res_final = nisa.tensor_reduce(
      np.min, data=neg_max_res[ip_max, if_max],
      axis=(1,), dtype=kernel_dtype, negate=False)

    ip_softmax = nl.arange(q_seq_tile_size)[:, None]
    if_softmax = nl.arange(seqlen)[None, :]
    ip_sum_res = nl.arange(q_seq_tile_size)[:, None]
    if_sum_res = nl.arange(d_head_tile_size)[None, :]

    softmax_res = nl.ndarray((par_dim(q_seq_tile_size), seqlen), dtype=pe_in_dt)
    sum_divisor = nl.ndarray((par_dim(q_seq_tile_size), d_head_tile_size), dtype=kernel_dtype)

    # Simply use a large tile of seq_len in size since this is a "blocking" instruction
    # Assuming the compiler will merge exp and reduce_add into a single instruction on ACT
    exp_res = nisa.activation(np.exp,
                              data=qk_res_buf[ip_softmax, if_softmax],
                              bias=neg_max_res_final, scale=1.0)

    sum_res = nisa.tensor_reduce(np.add, data=exp_res, axis=(1,),
                          dtype=kernel_dtype)
    softmax_res[ip_softmax, if_softmax] = nl.copy(exp_res, dtype=pe_in_dt)

    sum_reciprocal_broadcast = (1.0 / sum_res).broadcast_to((q_seq_tile_size, d_head_tile_size))
    sum_divisor[ip_sum_res, if_sum_res] = nl.copy(sum_reciprocal_broadcast, dtype=kernel_dtype)

    # Buffer for transposed softmax results (FP32 in PSUM)
    trans_softmax_res = nl.ndarray(
      (par_dim(k_seq_tile_size), k_seq_n_tiles, q_seq_tile_size),
      dtype=pe_in_dt)

    # Result psum buffer has the hidden dim as P
    attn_res_psum = nl.zeros((par_dim(d_head_tile_size), q_seq_tile_size),
                             dtype=np.float32, buffer=nl.psum)

    ip_scores_t = nl.arange(k_seq_tile_size)[:, None]
    if_scores_t = nl.arange(q_seq_tile_size)[None, :]
    # Loop over matmul_1 contraction
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
      ###################################
      # Step 5. transpose(softmax_res)
      ###################################
      ip_scores = nl.arange(q_seq_tile_size)[:, None]
      if_scores = nl.arange(k_seq_tile_size)[None, :]

      trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t] = nisa.nc_transpose(
        softmax_res[ip_scores, i_k_seq_tile * k_seq_tile_size + if_scores])

    ip_out = nl.arange(d_head_tile_size)[:, None]
    if_out = nl.arange(q_seq_tile_size)[None, :]
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
      ######################################################################
      # Step 6. matmul_1(stationary=trans_v, moving=trans_softmax_res, contract=seqlen_v=seqlen_k)
      ######################################################################
      ip_v_t = nl.arange(k_seq_tile_size)[:, None]
      if_v_t = nl.arange(d_head_tile_size)[None, :]
      attn_res_psum[ip_out, if_out] += \
        nisa.nc_matmul(moving=trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t],
                       stationary=trans_v[ip_v_t, i_k_seq_tile, if_v_t])

    attn_res_sbuf = nl.copy(attn_res_psum[ip_out, if_out], dtype=kernel_dtype)

    attn_res_div = attn_res_sbuf * nisa.nc_transpose(sum_divisor[ip_sum_res, if_sum_res])

    nl.store(
      out_ref[batch_id, i_q_seq_tile * q_seq_tile_size + if_out, ip_out],
      value=attn_res_div)

  return out_ref
