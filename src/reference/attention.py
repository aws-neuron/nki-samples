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
  training:bool = True
  should_transpose_v:bool = False

  __annotations__ = {
    'seq_tile_size': int,
    'training': bool,
    'should_transpose_v': bool
  }

def _flash_attention_core(q_local_tile, k, v,
                          q_h_per_k_h, seqlen_q, nheads,
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
                  + batch_id * nheads * seq_k_num_tiles * seq_q_num_tiles
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


@nki.jit
def flash_fwd(q, k, v, seed,
              softmax_scale=None,
              use_causal_mask=True,
              mixed_precision=True,
              dropout_p=0.0, config=None):
  """
  Flash Attention Forward kernel

  IO tensor layouts:
    - q: shape   (bs, n_heads, d, seq_q)
    - k: shape   (bs, nk_heads, d, seq_k)
    - v: shape   (bs, nv_heads, d, seq_v) if config.should_transpose_v  else (bs, nv_heads, seq_v, d)
    - seed: shape (1,)
    - o: shape (bs, n_heads, seq_q, d)
    - lse: shape (bs, n_heads, nl.tile_size.pmax, seq // nl.tile_size.pmax) if training else None
    - This kernel requires seq_k == seq_v

  IO tensor dtypes:
    - This kernel assumes all IO tensors have the same dtype
    - If mixed_percision is True, then all Tensor Engine operation will be performed in
      bfloat16 and accumulation will be performed in float32. Otherwise the intermediates
      will be in the same type as the inputs.

  Compile-time Constants:
    - softmax_scale: scaling for softmax, is None, default is `1.0/(d**0.5)`
    - mixed_precision: flag to set non-matmul ops in fp32 precision, defualt is set to `true`, if false, we use same precision as input types
    - causal_mask: flag to set causal masking
    - config: Instance of dataclass :class:`nki.kernels.attention.FlashConfig` with Performance config parameters for flash attention with default values
        seq_tile_size: `default=2048`, size of the kv tile size for attention computation reduction
        training: bool to indicate training vs inference `default=True`

  Performance Notes:
    For better performance, the kernel is tiled to be of size `LARGE_TILE_SZ`, and Flash attention math techniques are applied in unit
    of `LARGE_TILE_SZ`. Seqlen that is not divisible by `LARGE_TILE_SZ` is not supported at the moment.

  GQA support Notes:
    the spmd kernel for launching kernel should be on kv_heads instead of nheads

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
  kernel_dtype = nl.bfloat16 if mixed_precision else q.dtype
  acc_type = np.dtype(np.float32) if mixed_precision else kernel_dtype

  o = nl.ndarray((b, h, seqlen_q, d), dtype=q.dtype, buffer=nl.shared_hbm)
  if config.training:
    lse = nl.ndarray((b, h, nl.tile_size.pmax, seqlen_q // nl.tile_size.pmax),
                     dtype=acc_type, buffer=nl.shared_hbm)
  else:
    lse = None

  i_q_p = nl.arange(B_P_SIZE)[:,None]
  i_0_f = nl.arange(1)[None, :]

  batch_id = nl.program_id(axis=0)

  head_dims = list(range(1, nl.program_ndim()))
  head_dims_shape = list(nl.num_programs(i) for i in head_dims)
  head_dims_idx = list(nl.program_id(i) for i in head_dims)
  head_id = linearize(head_dims_shape, head_dims_idx)

  softmax_scale = softmax_scale or (1.0 / (d ** 0.5))

  n_tile_q = seqlen_q // B_P_SIZE # since q will be loaded on tensor engine

  LARGE_TILE_SZ = config.seq_tile_size
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

  for i_q_h in nl.affine_range(q_h_per_k_h):

    # =============== Global Flash Attention accumulators ====================== #
    o_buffer = nl.full((n_tile_q, num_large_k_tile, par_dim(B_P_SIZE), d), 0.0, dtype=acc_type, buffer=nl.sbuf)
    l_buffer = nl.full((n_tile_q, num_large_k_tile, par_dim(B_P_SIZE), 1), 0.0, dtype=acc_type, buffer=nl.sbuf)
    m_buffer = nl.full((n_tile_q, num_large_k_tile, par_dim(B_P_SIZE), 1), 0.0, dtype=acc_type)
    # =============== Global Flash Attention accumulators END ================== #

    j = 0
    cur_k_tile = nl.ndarray((par_dim(B_D_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
    cur_v_tile = nl.ndarray((LARGE_TILE_SZ//B_P_SIZE, par_dim(B_P_SIZE), B_D_SIZE), dtype=kernel_dtype)
    load_tile_size = B_P_SIZE
    for k_i in nl.affine_range(LARGE_TILE_SZ // load_tile_size):
      load_p = nl.arange(B_D_SIZE)[:, None]
      load_f = nl.arange(load_tile_size)[None, :]
      cur_k_tile[load_p, load_tile_size*k_i+load_f] = nl.load(
        k[batch_id, head_id, load_p, load_tile_size*k_i+load_f]
      )
    if config.should_transpose_v:
      for v_i in nl.affine_range(LARGE_TILE_SZ // load_tile_size):
        load_p = nl.arange(B_D_SIZE)[:, None]
        load_f = nl.arange(B_P_SIZE)[None, :]

        loaded = nl.load(v[batch_id, head_id, load_p, B_P_SIZE*v_i+load_f], dtype=kernel_dtype)
        store_p = nl.arange(B_P_SIZE)[:, None]
        store_f = nl.arange(B_D_SIZE)[None, :]
        cur_v_tile[v_i, store_p, store_f] = nisa.nc_transpose(loaded)
    else:
      for v_i in nl.affine_range(LARGE_TILE_SZ // load_tile_size):
        load_p = nl.arange(B_P_SIZE)[:, None]
        load_f = nl.arange(B_D_SIZE)[None, :]

        cur_v_tile[v_i, load_p, load_f] = nl.load(v[batch_id, head_id, B_P_SIZE*v_i+load_p, load_f], dtype=kernel_dtype)

    for i in nl.affine_range(n_tile_q):
      i_f_128 = nl.arange(B_P_SIZE)[None, :]
      i_f_d = nl.arange(B_D_SIZE)[None, :]
      i_p_d = nl.arange(B_D_SIZE)[:,None]
      q_tile = nl.ndarray((B_D_SIZE, B_P_SIZE),dtype=kernel_dtype)
      q_tile[i_p_d, i_f_128] = nl.load(q[batch_id,
                                        head_id * q_h_per_k_h + i_q_h, i_p_d,
                                        i * B_P_SIZE + i_f_128],
                                      dtype=kernel_dtype) * softmax_scale # load (d, 128) tile in SBUF
      # handle first tile and compute max and lse explicitly by passing initialize=True
      _flash_attention_core(q_local_tile=q_tile, k=cur_k_tile, v=cur_v_tile,
                            q_h_per_k_h=q_h_per_k_h, seqlen_q=seqlen_q, nheads=h,
                            o_buffer=o_buffer[i], l_buffer=l_buffer[i], m_buffer=m_buffer[i],
                            batch_id=batch_id, head_id=head_id,
                            gqa_head_idx=i_q_h, q_tile_idx=i, local_k_large_tile_idx=0,
                            kernel_dtype=kernel_dtype, acc_type=acc_type,
                            flash_config=config, use_causal_mask=use_causal_mask,
                            initialize=True,
                            B_P_SIZE=B_P_SIZE, B_F_SIZE=B_F_SIZE, B_D_SIZE=B_D_SIZE,
                            dropout_p=dropout_p, dropout_p_tensor=dropout_p_tensor, seed_tensor=seed_local)

    for j in nl.sequential_range(1, num_large_k_tile):
      cur_k_tile = nl.ndarray((par_dim(B_D_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
      cur_v_tile = nl.ndarray((LARGE_TILE_SZ//B_P_SIZE, par_dim(B_P_SIZE), B_D_SIZE), dtype=kernel_dtype)
      load_tile_size = B_P_SIZE
      for k_i in nl.affine_range(LARGE_TILE_SZ // load_tile_size):
        load_p = nl.arange(B_D_SIZE)[:, None]
        load_f = nl.arange(load_tile_size)[None, :]
        cur_k_tile[load_p, load_tile_size*k_i+load_f] = nl.load(
          k[batch_id, head_id, load_p, j*LARGE_TILE_SZ+load_tile_size*k_i+load_f]
        )
      if config.should_transpose_v:
        for v_i in nl.affine_range(LARGE_TILE_SZ // load_tile_size):
          load_p = nl.arange(B_D_SIZE)[:, None]
          load_f = nl.arange(B_P_SIZE)[None, :]

          loaded = nl.load(v[batch_id, head_id, load_p, j*LARGE_TILE_SZ+B_P_SIZE*v_i+load_f], dtype=kernel_dtype)
          store_p = nl.arange(B_P_SIZE)[:, None]
          store_f = nl.arange(B_D_SIZE)[None, :]
          cur_v_tile[v_i, store_p, store_f] = nisa.nc_transpose(loaded)
      else:
        for v_i in nl.affine_range(LARGE_TILE_SZ // load_tile_size):
          load_p = nl.arange(B_P_SIZE)[:, None]
          load_f = nl.arange(B_D_SIZE)[None, :]

          cur_v_tile[v_i, load_p, load_f] = nl.load(v[batch_id, head_id, j*LARGE_TILE_SZ+B_P_SIZE*v_i+load_p, load_f], dtype=kernel_dtype)

      for i in nl.affine_range(n_tile_q):
        i_f_128 = nl.arange(B_P_SIZE)[None, :]
        i_f_d = nl.arange(B_D_SIZE)[None, :]
        i_p_d = nl.arange(B_D_SIZE)[:,None]
        q_tile = nl.ndarray((B_D_SIZE, B_P_SIZE),dtype=kernel_dtype)
        q_tile[i_p_d, i_f_128] = nl.load(q[batch_id,
                head_id * q_h_per_k_h + i_q_h, i_p_d,
                i * B_P_SIZE + i_f_128],
              dtype=kernel_dtype) * softmax_scale # load (d, 128) tile in SBUF
        _flash_attention_core(q_local_tile=q_tile, k=cur_k_tile, v=cur_v_tile,
                              q_h_per_k_h=q_h_per_k_h, seqlen_q=seqlen_q, nheads=h,
                              o_buffer=o_buffer[i], l_buffer=l_buffer[i], m_buffer=m_buffer[i],
                              batch_id=batch_id, head_id=head_id,
                              gqa_head_idx=i_q_h, q_tile_idx=i, local_k_large_tile_idx=j,
                              kernel_dtype=kernel_dtype, acc_type=acc_type,
                              flash_config=config, use_causal_mask=use_causal_mask,
                              initialize=False,
                              B_P_SIZE=B_P_SIZE, B_F_SIZE=B_F_SIZE, B_D_SIZE=B_D_SIZE,
                              dropout_p=dropout_p, dropout_p_tensor=dropout_p_tensor, seed_tensor=seed_local)

    # -------- write output to buffer on HBM ------------ #
    for i in nl.affine_range(n_tile_q):
      out = nl.ndarray((par_dim(B_P_SIZE), B_D_SIZE), dtype=kernel_dtype)
      out[i_q_p, i_f_d] = nl.multiply(o_buffer[i, num_large_k_tile - 1, i_q_p, i_f_d],
                                      nl.exp(m_buffer[i, num_large_k_tile - 1, i_q_p, i_0_f] - l_buffer[i, num_large_k_tile - 1, i_q_p, i_0_f]),
                                      dtype=kernel_dtype)

      nl.store(o[batch_id, head_id * q_h_per_k_h + i_q_h, i*B_P_SIZE + i_q_p, i_f_d], out[i_q_p, i_f_d])
      if not inference:
        lse_local = nl.zeros((par_dim(B_P_SIZE), 1), dtype=acc_type)
        lse_local[i_q_p, i_0_f] = nl.copy(l_buffer[i, num_large_k_tile - 1, i_q_p, i_0_f], dtype=acc_type)
        nl.store(lse[batch_id, head_id * q_h_per_k_h + i_q_h, i_q_p, i + i_0_f], lse_local[i_q_p, i_0_f])

  if config.training:
    return o, lse

  return o

@nki.jit
def flash_attn_bwd(
  q_ref, k_ref, v_ref, o_ref,
  dy_ref,
  lse_ref,
  seed_ref,
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
   - out_dq_ref: shape (bs, nheads, head_size, seq)
   - out_dk_ref: shape (bs, nheads, head_size, seq)
   - out_dv_ref: shape (bs, nheads, head_size, seq)

  Detailed steps:
    1. D = rowsum(dO ◦ O) (pointwise multiply)

    2. Recompute (softmax(Q^T@K))

      2.1 Q^T@K
      2.2 Scale the QK score
      2.3 Apply causal mask
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
  assert lse_ref.dtype == mixed_dtype

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

  # Different batch samples/attention heads have independent attention
  batch_id = nl.program_id(axis=0)

  head_dims = list(range(1, nl.program_ndim()))
  head_dims_shape = list(nl.num_programs(i) for i in head_dims)
  head_dims_idx = list(nl.program_id(i) for i in head_dims)
  head_id = linearize(head_dims_shape, head_dims_idx)

  assert n_elts(head_dims_shape) == nheads, \
    f"The grid shape mismatch, got {n_elts(head_dims_shape)} but should be {nheads}"

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
        seqlen_q=seqlen_q, seqlen_k=seqlen_k, d_head=d_head, nheads=nheads,
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
        out_dv_ref[batch_id, head_id,
                   i_d_head_tile * d_head_tile_size + ip_dkv,
                   i_k_seq_tile * k_seq_tile_size + if_dkv],
        value=dv_psum[i_d_head_tile, ip_dkv, if_dkv],
      )

      nl.store(
        out_dk_ref[batch_id, head_id,
                    i_d_head_tile * d_head_tile_size + ip_dkv,
                    i_k_seq_tile * k_seq_tile_size + if_dkv],
        value=dk_psum[i_d_head_tile, ip_dkv, if_dkv],
      )

  # Write dQ
  for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
    for i_d_head_tile in nl.affine_range(d_head_n_tiles):
      ip_dq = nl.arange(d_head_tile_size)[:, None]
      if_dq = nl.arange(q_seq_tile_size)[None, :]

      nl.store(
        out_dq_ref[batch_id, head_id,
                   i_d_head_tile * d_head_tile_size + ip_dq,
                   i_q_seq_tile * q_seq_tile_size + if_dq],
        value=dq_local_reduced[i_q_seq_tile, i_d_head_tile, ip_dq, if_dq],
      )

  return out_dq_ref, out_dk_ref, out_dv_ref


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
              + batch_id * nheads * k_seq_n_tiles * q_seq_n_tiles
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


@nki.jit
def fused_self_attn_for_SD_small_head_size(q_ref, k_ref, v_ref, use_causal_mask=False,
                                           mixed_percision=True):
  """
  Fused self attention kernel for small head size Stable Diffusion workload.

  Computes softmax(QK^T)V. Decoder model can optionally include a causal mask
  application. Does not include QKV rojection, output projection, dropout,
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
   - If mixed_percision is True, then all Tensor Engine operation will be performed in
     bfloat16 and accumulation will be performed in float32. Otherwise the intermediates
     will be in the same type as the inputs.
  """
  # Use q_ref dtype as the intermediate tensor dtype
  # Assume all IO tensors have the same dtype
  kernel_dtype = q_ref.dtype
  pe_in_dt = nl.bfloat16 if mixed_percision else np.float32
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
