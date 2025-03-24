"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

Kernels written by the AWS Neuron.

WARNING: These kernels:
   - Are tested only against internal nightly builds
   - May not be compatible with public NeuronSDK releases
   - Have not been extensively tested across all input configurations
   - Carry no compatibility guarantees
   - The behavior of these kernels may be modified without prior notice

Attention Decode Kernel for token-generation.

This kernel is internal use only, and its behaviour might change in the future
without warning.

"""
import numpy as np
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki as nki
import neuronxcc.nki.typing as nt
from neuronxcc.nki import trace
from neuronxcc.nki._private.private_api import core_barrier, sendrecv

@trace(platform_target='trn2', show_compiler_tb=True, experimental_flags='skip-non-top-level-shared-hbm-check')
def attention_token_gen_kernel(Q, K, V, K_cache, V_cache, mask_cache, mask_active, out, K_cache_transposed=True,
                               QKV_in_sbuf=False, DBG_QK=None, DBG_QK_MAX=None, DBG_QK_EXP_T=None, DBG_EXP_SUM=None):
  #== Verify shapes.
  cache_shape = (K_cache.shape[0], K_cache.shape[2], K_cache.shape[1]) if K_cache_transposed else K_cache.shape
  B, S_max_ctx, d_head = cache_shape
  assert cache_shape == (B, S_max_ctx, d_head)
  assert cache_shape == V_cache.shape

  S_tkg, S_ctx = mask_cache.shape[-2:]

  if QKV_in_sbuf: # Expect [d, BnS]
    n_heads = Q.shape[1] // B // S_tkg
  else:
    _, _, n_heads, _ = Q.shape
    assert tuple(Q.shape) == (B, d_head, n_heads, S_tkg)
  assert tuple(K.shape) == (d_head, B * S_tkg) if QKV_in_sbuf else (B, d_head, S_tkg)
  # FIXME QKV_in_sbuf does not apply to V yet.
  # assert tuple(V.shape) == (S_tkg, B * d_head) if QKV_in_sbuf else (B, S_tkg, d_head)
  assert tuple(V.shape) == (B, S_tkg, d_head)
  assert S_tkg <= 128

  assert mask_cache.shape == (B, n_heads, S_tkg, S_ctx)
  assert mask_active.shape == (B, n_heads, S_tkg, S_tkg)

  # LNC2.
  grid_ndim = nl.program_ndim()
  assert grid_ndim == 0 or grid_ndim == 1, "attention_token_gen_kernel only supports no specialization or specialization along one axis"
  n_prgs, prg_id = (nl.num_programs(axes=0), nl.program_id(axis=0)) if grid_ndim != 0 else (1, 0)
  assert n_prgs <= 2, f'attention_token_gen_kernel supports unsharded or LNC-2 sharded; but got a spmd grid size of {n_prgs}'
  last_shard = prg_id == n_prgs - 1

  # We shard on S_ctx.
  assert S_ctx % n_prgs == 0, f"Expect context length to be divisible by number of shards of {n_prgs}."
  sharded_S_ctx = S_ctx // n_prgs
  assert sharded_S_ctx % 128 == 0, \
      f'attention_token_gen_kernel expects the sharded context length to be a multiple of 128, ' \
      f'but got context length of {S_ctx} with {n_prgs} shards.'
  sharded_S_ctx_offset = sharded_S_ctx * prg_id

  # For softmax, we pack (batch, num_heads, and S_tkg) together on partition dimension.
  nS = n_heads * S_tkg
  BnS = B * nS
  assert BnS <= 128, \
          'attention_token_gen_kernel expects batch_size * num_heads * decoding_sequence_length to be <= 128, ' \
          f'but got {B} * {n_heads} * {S_tkg}'

  # sendrecv pipe IDs:
  PIPE_ID_MM1_MAX = 1
  PIPE_ID_EXP_SUM = 2

  #  When transposing S_ctx onto partition dimension, we have to fold it into blocks.
  num_S_ctx_folds = sharded_S_ctx // 128
  num_S_max_ctx_folds = S_max_ctx // 128

  def prepQSbuf(Q):
    ''' Loasd Q tensor into SBUF and arrange in a layout that is friendly for partition packing '''

    if QKV_in_sbuf:
      Q_sb = Q
    else:
      # Load entire Q tensor into sbuf at once into [d_head, B * n_heads * S_tkg].
      Q_sb = nl.ndarray((d_head, BnS), dtype=Q.dtype, buffer=nl.sbuf)
      Q = Q.reshape((B, d_head, nS))
      # DMA copy with a 3D access pattern so we can load at once.
      i_p, i_b, i_f = nl.mgrid[0:d_head, 0:B, 0:nS]
      nisa.dma_copy(src=Q[i_b, i_p, i_f], dst=Q_sb[i_p, i_b * nS + i_f])

    # We want to pack QK results from all batches into partition dimension.
    # This will allow us to do softmax efficiently with all partitions done at once.
    #
    # The `Q @ K^T` MM is: [B, d_head, n_heads * S_tkg] @ [B, d_head, S_ctx] => [B, par_dim(n_heads * S_tkg), S_ctx]
    # `n_heads * S_tkg` is typically small. The softmax MM1 will need nS to be on partition dimension.
    # If we pack multiple batches of nS together on partition, the softmax of all batches can happen at once.
    #
    # To pack batches, during MM1, batch-X's Q head will be placed on a partition offset of `X * nS`.
    # However our PE architecture only allows offset at multiple of 32.
    # To work around this, For the MM of `Q[d_head, nS] @ K[d_head, moving_dim]`, we can change nS to BnS, by placing
    # the nS columns on the right offset, and padding zeros before and after.
    assert BnS <= 128
    Q_sb_padded = nl.zeros((d_head, B * BnS), dtype=Q.dtype, buffer=nl.sbuf)

    for b in nl.affine_range(B):
      offset = b*BnS + b*nS
      Q_sb_padded[0:d_head, offset : offset+nS] = Q_sb[0:d_head, b*nS : (b+1)*nS]

    return Q_sb_padded

  def prepCausalMask(mask_cache, mask_active):
    ''' Load Causal Masks into Sbuf with BnS on the partition dimension. '''
    ''' Overwrite the last S_tkg columns of mask with mask_active. '''
    mask_cache = mask_cache.reshape((BnS, S_ctx))
    causal_mask = nl.load(mask_cache[0:BnS, sharded_S_ctx_offset : sharded_S_ctx_offset + sharded_S_ctx])
    if last_shard & (S_tkg > 1):
      causal_mask[:, sharded_S_ctx - S_tkg:] = nl.load(mask_active.reshape((BnS, S_tkg)))

    if S_tkg == 1:
      causal_mask[:, sharded_S_ctx - 1:] = True

    return causal_mask

  def loadK(K_cache, K, tile_size):
    num_tile = (sharded_S_ctx + tile_size - 1) // tile_size  # Tile K_cache's S_ctx dimension into size of 2048.
    K_sb = nl.ndarray((num_tile, B, nl.par_dim(d_head), tile_size), dtype=K.dtype, buffer=nl.sbuf)

    K_cache = K_cache.reshape((B, d_head, S_max_ctx) if K_cache_transposed else (B, S_max_ctx, d_head))
    for mm1_tile_i in nl.static_range(num_tile):
      # Load K cache.
      for b in nl.static_range(B):
        # Load K_cache into sbuf.
        if K_cache_transposed: # Already transposed.
          i_tile = nl.arange(tile_size)[None, :] + mm1_tile_i * tile_size
          K_sb[mm1_tile_i][b] = nl.load(
              K_cache[b][nl.arange(d_head)[:, None], i_tile + sharded_S_ctx_offset], mask=(i_tile < sharded_S_ctx))
        else:
          i_tile = nl.arange(tile_size)[:, None] + mm1_tile_i * tile_size
          K_sb[mm1_tile_i][b] = nisa.dma_transpose(
              K_cache[b][i_tile + sharded_S_ctx_offset, nl.arange(d_head)[None, :]], mask=(i_tile < sharded_S_ctx))

        if last_shard & (mm1_tile_i == num_tile - 1):
          # FIXME: load all batches at once
          last_K_tile_tile_size = sharded_S_ctx - (num_tile - 1) * tile_size
          assert last_K_tile_tile_size > S_tkg
          K_sb[mm1_tile_i][b][:, last_K_tile_tile_size - S_tkg : last_K_tile_tile_size] = \
              nl.copy(K[:, b*S_tkg : (b+1)*S_tkg]) if QKV_in_sbuf else nl.load(K[b])
    return K_sb


  def MM1(Q_sb_padded, K_sb, causal_mask, tile_size):
    # Keeps the result of Q @ K.T
    QK_sb = nl.ndarray((BnS, sharded_S_ctx), dtype=np.float32, buffer=nl.sbuf)

    num_tile = (sharded_S_ctx + tile_size - 1) // tile_size  # Tile K_cache's S_ctx dimension into size of 2048.

    for mm1_tile_i in nl.static_range(num_tile):
      # Q @ K.T
      num_t512 = tile_size // 512 # Tile current tile tile into 512 for psum
      for t512_i in nl.affine_range(num_t512):
        QKc_psums = nl.zeros((BnS, 512), dtype=np.float32, buffer=nl.psum)

        offset_in_S_ctx = mm1_tile_i * tile_size + t512_i * 512
        i_K_p = nl.arange(d_head)[:, None]
        i_K_f = nl.arange(512)[None, :]

        # A separate MM for each batch, but all MMs write to the same PSUM space with each batch in its own partitions.
        for b in nl.affine_range(B):
          # The += below is not accumulation. Each batch's result is added to its own partitions that are still zeros.
          QKc_psums[:, 0:512] += nisa.nc_matmul(
              Q_sb_padded[:, b * BnS : (b+1) * BnS],
              K_sb[mm1_tile_i][b][i_K_p, i_K_f + t512_i * 512],
              mask = i_K_f + offset_in_S_ctx < sharded_S_ctx)

        # Copy to sbuf while applying the mask.
        QK_sb[:, offset_in_S_ctx : offset_in_S_ctx + 512] = \
            nl.where(
                condition = causal_mask[nl.arange(BnS)[:, None], i_K_f + offset_in_S_ctx],
                x = QKc_psums, # [nl.arange(BnS)[:,None], i_K_f],
                y = np.finfo(np.float32).min,
                mask = i_K_f + offset_in_S_ctx < sharded_S_ctx)

    if DBG_QK is not None:
      nl.store(DBG_QK[:, prg_id * sharded_S_ctx : (prg_id + 1) * sharded_S_ctx], QK_sb)

    return QK_sb

  def softmax(QK_sb):
    tile_size = 2048  # FIXME: try to make this bigger.
    num_tiles = (sharded_S_ctx + tile_size - 1) // tile_size  # Tile K_cache's S_ctx dimension into size of 2048.

    QK_exp_T = nl.ndarray((nl.par_dim(128), num_S_ctx_folds * BnS), dtype=Q.dtype, buffer=nl.sbuf)
    exp_sum = nl.ndarray((BnS, num_tiles), dtype=np.float32, buffer=nl.sbuf)

    QK_negate_max = nisa.tensor_reduce(np.max, data=QK_sb, axis=-1, dtype=np.float32, negate=True)
    if n_prgs > 1: # Use sendrecv to get the other core's QK_negate_max and do another Min reduce.
      QK_negate_max_recv = nl.zeros(QK_negate_max.shape, dtype=QK_negate_max.dtype)
      sendrecv(send_to_rank=(1 - prg_id), recv_from_rank=(1 - prg_id),
               src=QK_negate_max, dst=QK_negate_max_recv, pipe_id=PIPE_ID_MM1_MAX)
      QK_negate_max = nisa.tensor_tensor(QK_negate_max, QK_negate_max_recv, np.minimum)

    i_tile = nl.arange(tile_size)[None,:]
    for exp_tile_i in nl.affine_range(num_tiles):
      cur_tile_offset = exp_tile_i * tile_size

      # Apply exponential while accumulating the sum.
      QK_exp = nisa.activation_reduce(
          np.exp, QK_sb[nl.arange(BnS)[:, None], i_tile + cur_tile_offset],
          reduce_op=np.add, bias=QK_negate_max, reduce_res=exp_sum[:, exp_tile_i],
          mask=(i_tile + cur_tile_offset < sharded_S_ctx))

      # Transpose.
      num_t128 = tile_size // 128
      for i in nl.affine_range(num_t128):
        # i_f = nl.arange(128)[None,:] + (exp_tile_i * num_t128 + i) * 128
        i_f = nl.arange(128)[None,:] + i * 128
        QK_exp_T[nl.arange(128)[:,None], nl.arange(BnS)[None,:] + (exp_tile_i * num_t128 + i) * BnS] = \
            nl.transpose(QK_exp[nl.arange(BnS)[:,None], i_f], mask=(i_f + cur_tile_offset < sharded_S_ctx))

    if DBG_QK_MAX is not None:
      nl.store(DBG_QK_MAX, QK_negate_max)
    if DBG_QK_EXP_T is not None:
      nl.store(DBG_QK_EXP_T[:, prg_id * (sharded_S_ctx // 128) * BnS : (prg_id + 1) * (sharded_S_ctx // 128) * BnS], QK_exp_T)

    return QK_exp_T, exp_sum

  #== Tiling logic for loading V cache and MM2.
  def find_closest_divisor(X, preferred_val, min_val):
    if X <= min_val:  # Cannot find a divisor of X that is bigger than min_val.
      return X
    # Get all divisors of X that are >= min_val, and find the closest to preferred_val.
    divisors = [i for i in range(min_val, X + 1) if X % i == 0]
    return min(divisors, key=lambda x: abs(x - preferred_val))

  num_folds_per_tile = find_closest_divisor(num_S_ctx_folds, num_S_ctx_folds // 4, 4)
  num_mm2_tiles = num_S_ctx_folds // num_folds_per_tile
  assert num_mm2_tiles * num_folds_per_tile == num_S_ctx_folds

  def loadV(V_cache, V):
    Vc_sb = nl.ndarray((B, num_mm2_tiles, nl.par_dim(128), num_folds_per_tile * d_head), dtype=V.dtype, buffer=nl.sbuf)
    V_cache = V_cache.reshape((num_S_max_ctx_folds * B, 128, d_head))
    for b in nl.static_range(B):
      for t in nl.static_range(num_mm2_tiles):
        # Load V cache.
        i_p, i_b, i_f = nl.mgrid[0:128, 0:num_folds_per_tile, 0:d_head]
        nisa.dma_copy(
            src=V_cache[i_b + num_S_max_ctx_folds * b + num_S_ctx_folds * prg_id + num_folds_per_tile * t, i_p, i_f],
            dst=Vc_sb[b][t][i_p, i_b * d_head + i_f])

        if last_shard & (t == num_mm2_tiles - 1):
          Vc_sb[b][t][nl.arange(S_tkg)[:, None] + 128 - S_tkg,
                      nl.arange(d_head)[None, :] + (num_folds_per_tile - 1) * d_head] = \
                          nl.load(V[b])
                          # FIXME QKV_in_sbuf does not apply to V yet because dst par may not be a multiple of 32.
                          # nl.copy(V[:, b*d_head : (b+1)*d_head]) if QKV_in_sbuf else nl.load(V[b])
    return Vc_sb


  def MM2(QK_exp_T, Vc_sb):
    mm2_psum = nl.zeros((d_head, BnS), dtype=nl.float32, buffer=nl.psum)
    for b in nl.affine_range(B):
      for t in nl.affine_range(num_mm2_tiles):
        for mm2_t128_i in nl.affine_range(num_folds_per_tile):
          i_p128 = nl.arange(128)[:, None]
          cur_offset = mm2_t128_i * 128

          i_V_f = nl.arange(d_head)[None, :] + mm2_t128_i * d_head
          i_QK_T_f = nl.arange(n_heads * S_tkg)[None,:] + (t * num_folds_per_tile + mm2_t128_i) * BnS + b * nS

          V_tile = Vc_sb[b][t][i_p128, i_V_f]
          QK_exp_tile = QK_exp_T[i_p128, i_QK_T_f]

          mm2_psum[:, b * nS : (b+1) * nS] += nisa.nc_matmul(
                  V_tile, QK_exp_tile, mask=((mm2_t128_i + t * num_folds_per_tile) < num_S_ctx_folds))
    mm2_sbuf = nl.copy(mm2_psum)
    return mm2_sbuf

  def fusedLoadVandMM2(QK_exp_T, V_cache, V):
    mm2_psum = nl.zeros((d_head, BnS), dtype=nl.float32, buffer=nl.psum)

    V_cache = V_cache.reshape((num_S_max_ctx_folds * B, 128, d_head))
    for b in nl.static_range(B):
      for t in nl.static_range(num_mm2_tiles):
        # Load V cache.
        Vc_sb = nl.ndarray((nl.par_dim(128), num_folds_per_tile * d_head), dtype=V.dtype, buffer=nl.sbuf)
        i_p, i_b, i_f = nl.mgrid[0:128, 0:num_folds_per_tile, 0:d_head]
        nisa.dma_copy(
            src=V_cache[i_b + num_S_ctx_folds * (b * n_prgs + prg_id) + num_folds_per_tile * t, i_p, i_f],
            dst=Vc_sb[i_p, i_b * d_head + i_f])

        if last_shard & (t == num_mm2_tiles - 1):
          Vc_sb[nl.arange(S_tkg)[:, None] + 128 - S_tkg,
                nl.arange(d_head)[None, :] + (num_folds_per_tile - 1) * d_head] = \
                        nl.load(V[b])
                        # FIXME QKV_in_sbuf does not apply to V yet because dst par may not be a multiple of 32.
                        # nl.copy(V[:, b*d_head : (b+1)*d_head]) if QKV_in_sbuf else nl.load(V[b])

        # MM2.
        for mm2_t128_i in nl.affine_range(num_folds_per_tile):
          i_p128 = nl.arange(128)[:, None]
          cur_offset = mm2_t128_i * 128

          i_V_f = nl.arange(d_head)[None, :] + mm2_t128_i * d_head
          i_QK_T_f = nl.arange(n_heads * S_tkg)[None,:] + (t * num_folds_per_tile + mm2_t128_i) * BnS + b * nS

          V_tile = Vc_sb[i_p128, i_V_f]
          QK_exp_tile = QK_exp_T[i_p128, i_QK_T_f]

          mm2_psum[:, b * nS : (b+1) * nS] += nisa.nc_matmul(
                  V_tile, QK_exp_tile, mask=((mm2_t128_i + t * num_folds_per_tile) < num_S_ctx_folds))

    mm2_sbuf = nl.copy(mm2_psum)
    return mm2_sbuf

  def divExpSum(mm2_sbuf, exp_sum):
    # Apply reciprocal at once for all batches.
    exp_sum = nisa.tensor_reduce(np.add, data=exp_sum, axis=-1, dtype=exp_sum.dtype)

    if n_prgs > 1: # Use sendrecv to get the other core's exp_sum and do another sum.
      exp_sum_recv = nl.zeros(exp_sum.shape, dtype=exp_sum.dtype)
      sendrecv(send_to_rank=(1 - prg_id), recv_from_rank=(1 - prg_id),
               src=exp_sum, dst=exp_sum_recv, pipe_id=PIPE_ID_EXP_SUM)
      exp_sum = nisa.tensor_tensor(exp_sum, exp_sum_recv, np.add)

    if DBG_EXP_SUM is not None:
      nl.store(DBG_EXP_SUM, exp_sum)

    exp_sum = nl.transpose(exp_sum).broadcast_to(mm2_sbuf.shape)
    return nl.divide(mm2_sbuf, exp_sum)

  def reduceMM2(mm2_div_sbuf, out):
    out = out.reshape((d_head, B * n_heads * S_tkg))
    if n_prgs == 1:
      nl.store(out, mm2_div_sbuf)
      return

    # If n_prgs > 1, need to reduce the mm2 result.
    # FIXME: use sendrecv?
    mm2_hbm = nl.ndarray((d_head, BnS * n_prgs), dtype=out.dtype, buffer=nl.shared_hbm)
    nl.store(mm2_hbm[:, prg_id * BnS : (prg_id + 1) * BnS], mm2_div_sbuf)
    core_barrier(mm2_hbm, tuple(range(n_prgs)))

    # if last_shard: # only one core needs to do the FMA.
    # Cannot do one core only store, got 'AssertionError: Output float16 %out(128, 4, 5) has no store def'
    nisa._tiled_offloaded_fma(mm2_hbm[:, 0:BnS], mm2_hbm[:, BnS:2*BnS], dst=out, scales=[1.0, 1.0])


  # Main function level.
  mm1_tile_size = 8192

  useFusedLoadVandMM2 = False  # Two flavours, not much perf difference. Fused with big tiles has better perf.

  Q_sb_padded = prepQSbuf(Q)
  causal_mask = prepCausalMask(mask_cache, mask_active)
  if useFusedLoadVandMM2:
    K_sb = loadK(K_cache, K, mm1_tile_size)
    QK_sb = MM1(Q_sb_padded, K_sb, causal_mask, mm1_tile_size)
    QK_exp_T, exp_sum = softmax(QK_sb)
    mm2_sbuf = fusedLoadVandMM2(QK_exp_T, V_cache, V)
  else:
    K_sb = loadK(K_cache, K, mm1_tile_size)
    Vc_sb = loadV(V_cache, V)
    QK_sb = MM1(Q_sb_padded, K_sb, causal_mask, mm1_tile_size)
    QK_exp_T, exp_sum = softmax(QK_sb)
    mm2_sbuf = MM2(QK_exp_T, Vc_sb)

  mm2_div_sbuf = divExpSum(mm2_sbuf, exp_sum)

  reduceMM2(mm2_div_sbuf, out)


@nki.jit(platform_target='trn2', show_compiler_tb=True,
         experimental_flags='enable-mutable-parameter, skip-mutable-return-check, skip-non-top-level-shared-hbm-check')
def llama3_attention_block_token_gen_kernel(
    X, W_qkv, W_gamma, rmsnorm_eps, cos, sin,
    K_cache: nt.tensor[nt.mutable], V_cache: nt.tensor[nt.mutable],
    mask_cache, mask_active, position_ids,
    update_cache, K_cache_transposed=True, fused_rmsnorm=True, DBG=False):
  '''
  Implements the entire attention block in llama3 for token-generation graph.
  Includes QKV projection, RoPE, attention, and output projection.
  Expected layout of input and output tensors:
  - X: [B, S_tkg, H]
  - W_qkv: [H, nd], where n is the number of heads, concatenating one or more Q heads, one K head and one V head.
  - W_gamma: [1, H], layer normalization weights in QKV projection.
  - cos, sin: [d // 2, S_tkg]
  - K_cache: [B, d, S_max_ctx] if K_cache_transposed else [B, S_max_ctx, d]
  - V_cache: [B, S_max_ctx, d]
  - mask_cache: [B, n, S_tkg, S_ctx]
  - mask_active: [B, n, S_tkg, S_tkg]
  - position_ids: [B, 1]

  Returns output in [d, B, n, S_tkg]

  '''

  def squeeze_head_dim(K_cache, V_cache):
    ''' Remove the head dimension if layout in BNSd or BNdS '''
    if len(K_cache.shape) != 4:
      assert len(K_cache.shape) == len(V_cache.shape) == 3
      return K_cache, V_cache, False

    assert len(V_cache.shape) == 4
    assert K_cache.shape[1] == V_cache.shape[1] == 1
    K_shape = (K_cache.shape[0], K_cache.shape[2], K_cache.shape[3])
    V_shape = (V_cache.shape[0], V_cache.shape[2], V_cache.shape[3])
    return K_cache.reshape(K_shape), V_cache.reshape(V_shape), True

  def unsqueeze_head_dim(K_cache, V_cache, cache_had_head_dim):
    ''' Add back the head dimension if original layout in BNSd or BNdS '''
    if not cache_had_head_dim:
      return K_cache, V_cache

    K_shape = (K_cache.shape[0], 1, K_cache.shape[1], K_cache.shape[2])
    V_shape = (V_cache.shape[0], 1, V_cache.shape[1], V_cache.shape[2])
    return K_cache.reshape(K_shape), V_cache.reshape(V_shape)

  K_cache, V_cache, cache_had_head_dim = squeeze_head_dim(K_cache, V_cache)


  ## Shape checking.
  B, S_tkg, H = X.shape
  _, S_max_ctx, d_head = V_cache.shape
  S_ctx = mask_cache.shape[-1]

  assert V_cache.shape[0] == B
  assert tuple(K_cache.shape) == ((B, d_head, S_max_ctx) if K_cache_transposed else (B, S_max_ctx, d_head))

  assert W_qkv.shape[1] % d_head == 0, "Expect QKV projection weights be packed together as (n_heads + 1 + 1) * d_head"
  n_heads = W_qkv.shape[1] // d_head - 2
  if fused_rmsnorm:
    assert tuple(W_gamma.shape) == (1, H)

  half_d = d_head // 2
  assert tuple(cos.shape) == (half_d, B, S_tkg)
  assert tuple(sin.shape) == (half_d, B, S_tkg)

  assert tuple(mask_cache.shape) == (B, n_heads, S_tkg, S_ctx)
  assert tuple(mask_active.shape) == (B, n_heads, S_tkg, S_tkg)

  # LNC2.
  grid_ndim = nl.program_ndim()
  assert grid_ndim == 0 or grid_ndim == 1, "llama3_attention_block_token_gen_kernel only supports no specialization or specialization along one axis"
  n_prgs, prg_id = (nl.num_programs(axes=0), nl.program_id(axis=0)) if grid_ndim != 0 else (1, 0)
  assert n_prgs <= 2, f'llama3_attention_block_token_gen_kernel supports unsharded or LNC-2 sharded; but got a spmd grid size of {n_prgs}'

  def QK_RoPE(QKV_out, cos, sin):
    '''
    Perform RoPE on Q and K heads.
    Do RoPE on both cores without sharding since compute is fast and sharding can incur synchronization overhead.
    Transpose Q and K heads. QKV_out is in nBSd layout.
      For each head in n,
        We load transpose into dBS layout.
        Do RoPE for all batches at once into dBS, and save into dnBS layout.
    '''
    from neuronxcc.nki._private_kernels.RoPE import RoPE_sbuf

    # To do RoPE at once for all batches, we can broadcast the cos/sin coeffs to multiple batches.
    cos_sbuf = nl.load(cos.reshape((half_d, B * S_tkg)))
    sin_sbuf = nl.load(sin.reshape((half_d, B * S_tkg)))

    QKV_n_BS_d = QKV_out.reshape((n_heads + 2, B * S_tkg, d_head))

    Q_RoPE = nl.ndarray((d_head, B * n_heads * S_tkg), dtype=X.dtype, buffer=nl.sbuf)
    for h in nl.affine_range(n_heads):
      Q_h_tp_d_BS = nisa.dma_transpose(QKV_n_BS_d[h])
      Q_h_tp_d_BS_RoPE = nl.ndarray(Q_h_tp_d_BS.shape, dtype=X.dtype, buffer=nl.sbuf)
      RoPE_sbuf(x_in_sb=Q_h_tp_d_BS, cos_sb=cos_sbuf, sin_sb=sin_sbuf, x_out_sb=Q_h_tp_d_BS_RoPE)
      # FIXME: can we do striped copy?
      for b in nl.affine_range(B):
        Q_RoPE[nl.arange(d_head)[:, None], nl.arange(S_tkg)[None, :] + b * n_heads * S_tkg + h * S_tkg] = nl.copy(Q_h_tp_d_BS_RoPE[nl.arange(d_head)[:, None], nl.arange(S_tkg)[None, :] + b * S_tkg])

    K_RoPE = nl.ndarray((d_head, B * S_tkg), dtype=X.dtype, buffer=nl.sbuf)
    K_tp_d_BS = nisa.dma_transpose(QKV_n_BS_d[n_heads])
    RoPE_sbuf(x_in_sb=K_tp_d_BS, cos_sb=cos_sbuf, sin_sb=sin_sbuf, x_out_sb=K_RoPE)

    return Q_RoPE, K_RoPE

  def update_KV_cache(K_cache, V_cache, K_RoPE, V, position_ids):
    assert position_ids.shape == (B, S_tkg)
    pos_id_sb = nl.load_transpose2d(position_ids)
    assert pos_id_sb.shape == (S_tkg, B)

    # Update V cache.
    assert V.shape == (S_tkg, B * d_head)
    assert V_cache.shape == (B, S_max_ctx, d_head)
    if n_prgs != 2 or prg_id == 0:
      for b in nl.static_range(B):
        # Use vector DGE.
        pos_id = pos_id_sb[nl.arange(S_tkg)[:, None], b]
        nl.store(V_cache[b][pos_id, nl.arange(d_head)[None, :]], V[:, b*d_head : (b+1)*d_head])


    # Update K cache.
    if n_prgs != 2 or prg_id == 1:
      assert K_RoPE.shape == (d_head, B * S_tkg)
      if K_cache_transposed:
        assert K_cache.shape == (B, d_head, S_max_ctx)
        for b in nl.affine_range(B):
          # Use scalar DGE.
          nl.store(K_cache[b][nl.arange(d_head)[:, None], nl.arange(S_tkg)[None, :] + pos_id_sb[0, b]],
                   K_RoPE[nl.arange(d_head)[:, None], nl.arange(S_tkg)[None, :] + b * S_tkg])
      else:
        assert K_cache.shape == (B, S_max_ctx, d_head)
        K_RoPE_tp = nl.transpose(K_RoPE) # (B * S_tkg, d_head)
        for b in nl.affine_range(B):
          # Use vector DGE.
          pos_id = pos_id_sb[nl.arange(S_tkg)[:, None], b]
          nl.store(K_cache[b][pos_id, nl.arange(d_head)[None, :]], K_RoPE_tp[b*S_tkg : (b+1)*S_tkg, :])

    return K_cache, V_cache

  # Main function level.

  # QKV projection, result in nBSd.
  QKV_out = nl.ndarray((n_heads + 2, B, S_tkg, d_head), dtype=X.dtype, buffer=nl.shared_hbm)

  from neuronxcc.nki._private_kernels.qkv import rmsnorm_qkv_isa_kernel
  rmsnorm_qkv_isa_kernel(lhs=X, rhs=W_qkv, ln_w=W_gamma, out=QKV_out, kernel_name="QKV", eps=rmsnorm_eps, fused_rmsnorm=fused_rmsnorm)

  if n_prgs > 1:
    core_barrier(QKV_out, tuple(range(n_prgs)))

  # RoPE on Q and K, result in dBnS, where n dimensions are num_heads & 1/squeezed-out for Q & K respectively.
  Q_RoPE, K_RoPE = QK_RoPE(QKV_out, cos, sin)

  # Divide Q by sqrt(d_head)
  Q_RoPE = nisa.tensor_scalar(data=Q_RoPE, op0=nl.multiply, operand0=(1/np.sqrt(d_head)), dtype=Q_RoPE.dtype)

  # Intermediate values for debugging.
  if DBG:
    DBG_Q_RoPE = nl.ndarray((d_head, B, n_heads, S_tkg), dtype=X.dtype, buffer=nl.shared_hbm)
    DBG_K_RoPE = nl.ndarray((d_head, B, S_tkg), dtype=X.dtype, buffer=nl.shared_hbm)
    nl.store(DBG_Q_RoPE.reshape(Q_RoPE.shape), Q_RoPE)
    nl.store(DBG_K_RoPE.reshape(K_RoPE.shape), K_RoPE)

  # Attention tkg.
  # Load V tensor into sbuf at once into [S_tkg, B * d_head].
  V_hbm = QKV_out[n_heads + 1]
  V_sbuf = nl.ndarray((S_tkg, B * d_head), dtype=X.dtype, buffer=nl.sbuf)
  i_p, i_b, i_f = nl.mgrid[0:S_tkg, 0:B, 0:d_head]
  nisa.dma_copy(src=V_hbm[i_b, i_p, i_f], dst=V_sbuf[i_p, i_b * d_head + i_f])

  if DBG:
    BnS = B * n_heads * S_tkg
    DBG_QK = nl.ndarray((BnS, S_ctx), dtype=np.float32, buffer=nl.shared_hbm)
    DBG_QK_MAX = nl.ndarray((BnS, 1), dtype=np.float32, buffer=nl.shared_hbm)
    DBG_QK_EXP_T = nl.ndarray((128, S_ctx // 128 * BnS), dtype=X.dtype, buffer=nl.shared_hbm)
    DBG_EXP_SUM = nl.ndarray((BnS, 1), dtype=np.float32, buffer=nl.shared_hbm)
  else:
    DBG_QK = DBG_QK_MAX = DBG_QK_EXP_T = DBG_EXP_SUM = None

  out = nl.ndarray((d_head, B, n_heads * S_tkg), dtype=X.dtype, buffer=nl.shared_hbm)
  attention_token_gen_kernel(Q_RoPE, K_RoPE, V_hbm, K_cache, V_cache, mask_cache, mask_active, out, K_cache_transposed,
                             QKV_in_sbuf=True, DBG_QK=DBG_QK, DBG_QK_MAX=DBG_QK_MAX, DBG_QK_EXP_T=DBG_QK_EXP_T, DBG_EXP_SUM=DBG_EXP_SUM)

  # Either return the new K/V heads, or return the updated K/V caches.
  if not update_cache:
    if n_prgs != 2 or prg_id == 0:
      new_K = nl.ndarray(K_RoPE.shape, dtype=X.dtype, buffer=nl.shared_hbm)
      nisa.dma_copy(dst=new_K, src=K_RoPE)
      new_K = new_K.reshape((d_head, B, S_tkg))

    if n_prgs != 2 or prg_id == 1:
      new_V = nl.ndarray(V_hbm.shape, dtype=X.dtype, buffer=nl.shared_hbm)
      nisa.dma_copy(dst=new_V, src=V_hbm)
      new_V = new_V.reshape((B, S_tkg, d_head))
  else:
    new_K, new_V = update_KV_cache(K_cache, V_cache, K_RoPE, V_sbuf, position_ids)
    new_K, new_V = unsqueeze_head_dim(new_K, new_V, cache_had_head_dim)

  if DBG:
    return out, new_K, new_V, QKV_out, DBG_Q_RoPE, DBG_K_RoPE, DBG_QK, DBG_QK_MAX, DBG_QK_EXP_T, DBG_EXP_SUM

  return out, new_K, new_V

