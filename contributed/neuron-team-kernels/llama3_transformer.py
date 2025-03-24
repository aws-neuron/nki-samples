"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

Kernels written by the AWS Neuron.

WARNING: These kernels:
   - Are tested only against internal nightly builds
   - May not be compatible with public NeuronSDK releases
   - Have not been extensively tested across all input configurations
   - Carry no compatibility guarantees
   - The behavior of these kernels may be modified without prior notice
"""

import numpy as np
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.nccl as nccl
from RoPE import RoPE
from neuronxcc.nki._private.private_api import \
    core_barrier, attention_kernel, qkv_kernel, qkv_fused_add_kernel, mlp_kernel, mlp_fused_add_kernel
from neuronxcc.nki._private_kernels import NormType

# Global SPMD grid variables used throughout the code.
n_prgs, prg_id = 1, 0

def __init_spmd_grid_size():
  """
  Initializes the spmd global variables n_prgs, prg_id
  """
  grid_ndim = nl.program_ndim()
  assert grid_ndim == 0 or grid_ndim == 1, \
      "llama3_transfomer_fwd_<tp|rmsnorm_sp> only supports no specialization or specialization along one axis"

  global n_prgs, prg_id
  if grid_ndim != 0 and nl.num_programs(axes=0) > 1:
    n_prgs = nl.num_programs(axes=0)
    prg_id = nl.program_id(axis=0)

def __core_barrier(data_tile):
  if n_prgs == 1:
    return
  core_barrier(data_tile, tuple(range(n_prgs)))

def transpose_and_RoPE(qkv_proj_out, cos, sin, bs, seqlen, DHead, DHead_offset, transposed_head, out, lnc_shard):
  """
  Transposes one of head (i.e., Q or K) in qkv_proj_out and apply RoPE

  Parameters:
    qkv_proj_out: hbm tensor holding QKV projection output. Expected to be in
      [bs, seqlen, DHead * N] shape, where N is due to several heads packed together (e.g., QQKV).
    cos, sin: RoPE coefficients in [DHead // 2, seqlen] shape.
    DHead_offset: indicates the offset of K head in the DHead * N dimension.
    transposed_head: hbm tensor to hold the transposed head.
    out: Transposed head with RoPE
    lnc_head: indicates if the compute in this function should be sharded.
  """
  local_n_prgs, local_prg_id = (n_prgs, prg_id) if lnc_shard else (1, 0)

  # We assume DHead is always 128, seqlen is always a multiple of 128 and tile along seqlen dimension.
  num_tiles_per_prg = seqlen // 128 // local_n_prgs
  assert num_tiles_per_prg * local_n_prgs * 128 == seqlen
  assert bs == 1 and len(qkv_proj_out.shape) == 3 and DHead == 128
  # Make sure transposed_head has the right shape.
  assert tuple(transposed_head.shape) == (bs, DHead, seqlen)

  # FIXME: add batch dimension to transpose.
  for _i in range(num_tiles_per_prg):
    # Each of the N shards takes a consecutive portion of seqlen dimension.
    # This is important because the sharding aligns with the sharding inside RoPE so we don't need core_barrier.
    i = local_prg_id * num_tiles_per_prg + _i

    head_memref = qkv_proj_out[0][nl.arange(128)[:, None] + i * 128, nl.arange(DHead)[None, :] + DHead_offset]
    transposed_memref = transposed_head[0][nl.arange(DHead)[:, None], nl.arange(128)[None, :] + i * 128]

    transposed_sbuf = nl.load_transpose2d(head_memref)
    nl.store(transposed_memref, transposed_sbuf)

  # Now do RoPE.
  for b in nl.static_range(bs):
    RoPE(transposed_head[b], cos, sin, out[b], lnc_shard)

  if lnc_shard:
    __core_barrier(out)


def matmul_o_proj(self_attn_out, W_o, o_proj):
  """
  Apply output projection.
  """
  B, K, M = self_attn_out.shape
  K_, N = W_o.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"
  assert tuple(o_proj.shape) == (B, M, N), f'{o_proj.shape} != [{B}, {M}, {N}]'

  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  TILE_K = nl.tile_size.pmax  # 128
  TILE_N = min(nl.tile_size.gemm_moving_fmax, N)  # 512

  # We are going to shard M, i.e., the seqlen dimension of LHS and output.
  assert M % (TILE_M * n_prgs) == 0
  assert K % TILE_K == 0
  assert N % TILE_N == 0

  Sharded_M = M // n_prgs

  # Load rhs, i.e., output projection weights.
  rhs = nl.ndarray((K // TILE_K, nl.par_dim(TILE_K), N), dtype=W_o.dtype, buffer=nl.sbuf)
  for k in nl.affine_range(K // TILE_K):
    rhs[k] = nl.load(W_o[k * TILE_K :(k + 1) * TILE_K, :])

  # Use affine_range to loop over tiles
  for b in nl.affine_range(B):
    # Load lhs, i.e., the self_attention_out
    lhs = nl.ndarray((K // TILE_K, nl.par_dim(TILE_K), Sharded_M), dtype=self_attn_out.dtype, buffer=nl.sbuf)
    for k in nl.affine_range(K // TILE_K):
      lhs[k] = nl.load(self_attn_out[b, k * TILE_K:(k + 1) * TILE_K, prg_id * Sharded_M : (prg_id + 1) * Sharded_M])

    for _m in nl.affine_range(Sharded_M // TILE_M):   # <-- shard M here.
      m = _m * n_prgs + prg_id
      for n in nl.affine_range(N // TILE_N):
        # Allocate a tensor in PSUM
        res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

        for k in nl.affine_range(K // TILE_K):
          # Accumulate partial-sums into PSUM
          res_psum += nl.matmul(
              lhs[k, :, _m * TILE_M:(_m + 1) * TILE_M],
              rhs[k, :, n * TILE_N:(n + 1) * TILE_N], transpose_x=True)

        # Copy the result from PSUM back to SBUF, and cast to expected output data-type
        res_sb = nl.copy(res_psum, dtype=o_proj.dtype)
        nl.store(o_proj[b, m * TILE_M : (m + 1) * TILE_M, n * TILE_N : (n + 1) * TILE_N], value=res_sb)


def sp_tile_fused_add_rmsnorm_all_gather(
    a_tensor, b_tensor, gamma_sbuf, out_tensor, tile_idx, tile_size, sp_tile_size, replica_groups):
  """
  Sequence parallel RMSNorm with fused add, followed by all-gather.
  a_tensor, b_tensor, and out_tensor should all be in [bs, sp_seqlen, D] shape.
  if b_tensor is provided, fused add will be performed and the summation result will be stored into b_tensor.
  """
  bs, sp_seqlen, D = a_tensor.shape
  assert b_tensor == None or b_tensor.shape == a_tensor.shape
  _, seqlen, _ = out_tensor.shape
  assert tuple(out_tensor.shape) == (bs, seqlen, D)

  # We will distribute the num_rmsnorm_tiles to LNCs.
  # So we want to make num_rmsnorm_tiles divisible by n_prgs.

  assert sp_tile_size % n_prgs == 0
  rmsnorm_tile_size = 128 if (sp_tile_size // n_prgs) > 128 else (sp_tile_size // n_prgs)
  num_rmsnorm_tiles = sp_tile_size // rmsnorm_tile_size

  assert rmsnorm_tile_size * num_rmsnorm_tiles == sp_tile_size

  i_D_2d = nl.arange(D)[None, :]

  # Each iteration computes a tile with size of [sp_tile_size, D].
  rmsnorm_result_hbm = nl.ndarray(shape = [bs, sp_tile_size, D], dtype = out_tensor.dtype, buffer=nl.shared_hbm)
  # An all-gather will be performed on current tile at the end.

  for b in nl.static_range(bs):
    for _rt in nl.static_range(num_rmsnorm_tiles // n_prgs):
      rt = _rt * n_prgs + prg_id
      # Each iteration computes rmsnorm_tile_size x D size of tensor.

      # Holding the a + b result in sbuf for current tile of [rmsnorm_tile_size, D]
      sum_result = nl.ndarray(shape=[rmsnorm_tile_size, D], dtype=out_tensor.dtype, buffer=nl.sbuf)

      # For slicing the seqlen dimension of [sp_seqlen, D] tensors.
      i_sl = nl.arange(rmsnorm_tile_size)[:, None] + tile_idx * sp_tile_size + rt * rmsnorm_tile_size

      a_ref = a_tensor[b][i_sl, i_D_2d]

      if b_tensor is not None:
        # FMA summation.
        b_ref = b_tensor[b][i_sl, i_D_2d]
        nisa._tiled_offloaded_fma(a_ref, b_ref, dst=sum_result.as_tile(), scales=[1.0, 1.0])
        nl.store(b_ref, sum_result)
      else:
        sum_result = nl.load(a_ref)

      # RMS Norm.
      rmsnorm_result = nl.rms_norm(x=sum_result, w=gamma_sbuf, axis=len(sum_result.shape)-1, n=D, epsilon=0)
      rmsnorm_result_hbm_ref = rmsnorm_result_hbm[b][nl.arange(rmsnorm_tile_size)[:, None] + rt * rmsnorm_tile_size, nl.arange(D)[None, :]]
      nl.store(rmsnorm_result_hbm_ref, rmsnorm_result)

  # Perform all-gather for current tile.
  assert sp_tile_size * len(replica_groups[0]) == tile_size, f'{sp_tile_size} * {len(replica_groups[0])} == {tile_size}'
  out_tensor_ref = out_tensor[nl.arange(bs)[:, None, None], nl.arange(tile_size)[None, :, None] + tile_idx * tile_size, nl.arange(D)[None, None, :]]
  nccl.all_gather(op=np.add, srcs=[rmsnorm_result_hbm], dsts=[out_tensor_ref],
                  replica_groups=replica_groups, all_gather_dim=0, dtype=out_tensor.dtype)

def tiled_rmsnorm_qkv(X, W_qkv, Gamma_qkv, qkv_proj_out, num_tiles, norm_type, fused_add_op0=None, fused_add_op1=None, eps = 1e-6):
  """
  Perform QKV projection with tiling along seqlen dimension.
  """
  # Tile along the seqlen dimension of X.
  bs, seqlen, D = X.shape
  _, qkv_dim = W_qkv.shape
  assert tuple(qkv_proj_out.shape) == (bs, seqlen, qkv_dim)

  tile_size = seqlen // num_tiles // n_prgs
  assert tile_size * num_tiles * n_prgs == seqlen, f'tile_size * num_tiles * n_prgs == seqlen: {tile_size} * {num_tiles} * {n_prgs} == {seqlen}'

  i_b = nl.arange(bs)[:, None, None]
  i_sl_t = nl.arange(tile_size)[None, :, None]
  i_x_D = nl.arange(D)[None, None, :]

  kernel_attrs = {}
  kernel_attrs["fusedRmsNorm"] = False if norm_type == NormType.NO_NORM else True
  kernel_attrs["norm_type"] = norm_type.value
  kernel_attrs["LncSize"] = n_prgs
  kernel_attrs["output_layout"] = 0 # QKVOutputLayout.BSD
  kernel_attrs["useTkgQKVKernel"] = False # This is CTE kernel

  for _t in nl.static_range(num_tiles):
    t = _t * n_prgs + prg_id
    X_ref = X[i_b, i_sl_t + t * tile_size, nl.arange(D)[None,None,:]]
    W_qkv_ref = W_qkv[nl.arange(D)[:,None], nl.arange(qkv_dim)[None,:]]
    Gamma_ref = Gamma_qkv[nl.arange(1)[:,None], nl.arange(D)[None,:]]
    out_ref = qkv_proj_out[i_b, i_sl_t + t * tile_size, nl.arange(qkv_dim)[None,None,:]]

    if not fused_add_op0: # No fused add.
      qkv_kernel("QKV", X_ref, W_qkv_ref, Gamma_ref, eps, out_ref, **kernel_attrs)
    else: # Fused add.
      assert norm_type == NormType.RMS_NORM, 'Fused add cannot be done without fused RMSNorm.'
      fa_op0_ref = fused_add_op0[i_b, i_sl_t + t * tile_size, nl.arange(D)[None,None,:]]
      fa_op1_ref = fused_add_op1[i_b, i_sl_t + t * tile_size, nl.arange(D)[None,None,:]]
      qkv_fused_add_kernel("QKV", X_ref, fa_op0_ref, fa_op1_ref, W_qkv_ref, Gamma_ref, eps, out_ref, **kernel_attrs)

  # Subsequent operations need the full QKV to work.
  # FIXME: transpose_k could start earlier, or fuse into here.
  __core_barrier(qkv_proj_out)

from collections import namedtuple
SpRmsNormParam = namedtuple('SpRmsNormParam', ['fused_add_sp', 'gamma', 'ag_rmsnorm'])
# Indicates whether tiles of reduce-scatter and all-gather should be interleaving or not.
INTERLEAVE_RS_AG = False

def attention_output_proj(qkv_proj_out, cos, sin, W_o, reduced_o_proj_out, attn_kernel_name, softmax_scale,
                          replica_groups, num_tiles, sp_rmsnorm_param=None):
  """
  Computes attention and output projection.
  """
  dtype = qkv_proj_out.dtype
  bs, seqlen, _ = qkv_proj_out.shape
  _, D = W_o.shape
  i_b = nl.arange(bs)[:, None, None]
  i_sl = nl.arange(seqlen)[None, :, None]
  i_x_D = nl.arange(D)[None, None, :]
  DHead = 128

  #==== Slice QKV projection output to get Q, K, and V.  K needs transpose.
  NUM_Q_HEADS = (qkv_proj_out.shape[-1] // 128) - 2
  assert (NUM_Q_HEADS + 2) * 128 == qkv_proj_out.shape[-1], \
      'qkv_proj_out.shape[-1] of {qkv_proj_out.shape[-1]} is not divisible by d_head of 128.'
  assert NUM_Q_HEADS == 1 or NUM_Q_HEADS == 2, 'NUM_Q_HEADS must be either 1 or 2.'
  assert NUM_Q_HEADS * 128 == W_o.shape[0]

  # Tile along the sequence length dimension for attention.
  assert seqlen % num_tiles == 0, f'Sequence length of {seqlen} is not divisible by {num_tiles} tiles.'
  tile_size = seqlen // num_tiles

  # Preload gamma into sbuf and use it for all tiles.
  if sp_rmsnorm_param:
    sp_seqlen = seqlen // len(replica_groups[0])
    sp_tile_size = sp_seqlen // num_tiles
    rmsnorm_tile_size = 128 if (sp_tile_size // n_prgs) > 128 else (sp_tile_size // n_prgs)
    g_tensor_sbuf = nl.load(sp_rmsnorm_param.gamma)
    gamma_sbuf = g_tensor_sbuf.broadcast_to([rmsnorm_tile_size, D])

  def rs(t): # Reduce-scatter
    assert reduced_o_proj_out.shape[1] * len(replica_groups[0]) == o_proj.shape[1] * num_tiles
    tiled_sp_num_rows = reduced_o_proj_out.shape[1] // num_tiles
    i_sl_t_sp = nl.arange(tiled_sp_num_rows)[None, :, None] + tiled_sp_num_rows * t
    reduced_o_proj_out_memref = reduced_o_proj_out[i_b, i_sl_t_sp, i_x_D]
    nccl.reduce_scatter(op=np.add, srcs=[o_proj], dsts=[reduced_o_proj_out_memref], replica_groups=replica_groups, reduce_scatter_dim=1, dtype=o_proj.dtype)

  def sp_rmsnorm_ag(t): # Sequence parallel RMSNorm, followed by all_gather.
    sp_tile_fused_add_rmsnorm_all_gather(
        reduced_o_proj_out, sp_rmsnorm_param.fused_add_sp, gamma_sbuf, sp_rmsnorm_param.ag_rmsnorm,
        t, tile_size, sp_tile_size, replica_groups)

  v_memref = qkv_proj_out[i_b, i_sl, nl.arange(DHead)[None, None, :] + (NUM_Q_HEADS + 1) * DHead]

  # K has to be transposed as expected by the attention kernel.
  transposed_k = nl.ndarray(shape = [bs, DHead, seqlen], dtype = dtype, buffer=nl.shared_hbm)
  k_RoPE = nl.ndarray(shape = [bs, DHead, seqlen], dtype = dtype, buffer=nl.shared_hbm)
  transpose_and_RoPE(qkv_proj_out, cos, sin, bs, seqlen, DHead, NUM_Q_HEADS * DHead, transposed_k, k_RoPE, lnc_shard=True)

  # Distribute Q heads amongst workers.
  num_q_heads_per_prg = NUM_Q_HEADS // n_prgs
  assert num_q_heads_per_prg * n_prgs == NUM_Q_HEADS, f'Current version requires NUM_Q_HEADS to be divisible by #. LNCs.'

  transposed_q = nl.ndarray(shape = [num_q_heads_per_prg, bs, DHead, seqlen], dtype = dtype, buffer=nl.hbm)
  q_RoPE       = nl.ndarray(shape = [num_q_heads_per_prg, bs, DHead, seqlen], dtype = dtype, buffer=nl.hbm)

  for local_h in nl.static_range(num_q_heads_per_prg):
    h = local_h * n_prgs + prg_id
    transpose_and_RoPE(qkv_proj_out, cos, sin, bs, seqlen, DHead, h * DHead, transposed_q[local_h], q_RoPE[local_h], lnc_shard=False)

  # affine_range: CC op not overlapped, two CC ops still happening at the
  #         end of attention and output projection.
  # static_range and sequential_range: CC op overlap happens (and 252us)
  # About affine_range, Tensorizer may try to parallelize the loop iterations and potentially
  # interleave operations between iterations and therefore prevent CC op overlapping.
  for t in nl.static_range(num_tiles):
    tile_seqlen_offset = tile_size * t
    i_sl_t = nl.arange(tile_size)[None, :, None] + tile_seqlen_offset

    self_attn_out = nl.ndarray([bs, DHead * NUM_Q_HEADS, tile_size], dtype = dtype, buffer=nl.shared_hbm)

    for local_h in nl.static_range(num_q_heads_per_prg):  # Distribute Q heads amongst NCs.
      h = local_h * n_prgs + prg_id
      self_attn_out_memref = self_attn_out[i_b, nl.arange(DHead)[None, :, None] + h * DHead, nl.arange(tile_size)[None, None, :]]

      q_memref = q_RoPE[local_h][i_b, nl.arange(DHead)[None, :, None], nl.arange(tile_size)[None, None, :] + t * tile_size]

      kernel_attrs = {
        "cache_softmax": False,
        "use_dma_transpose": False
      }
      attention_kernel(attn_kernel_name, q_memref, k_RoPE.as_tile(), v_memref, softmax_scale, self_attn_out_memref, **kernel_attrs)

    __core_barrier(self_attn_out)

    o_proj = nl.ndarray(shape = [bs, tile_size, D], dtype = dtype, buffer=nl.shared_hbm)

    assert W_o.shape[1] == D
    matmul_o_proj(self_attn_out, W_o, o_proj)

    if sp_rmsnorm_param: # SP RMSNorm
      rs(t)  # Do reduce_scatter
      if INTERLEAVE_RS_AG:
        sp_rmsnorm_ag(t) # If we interleave RS and AG, do SP-RMSNorm and all-gather right here.
    else: # Pure-TP, do an all-reduce here.
      reduced_o_proj_out_memref = reduced_o_proj_out[i_b, i_sl_t, i_x_D]
      nccl.all_reduce(op=np.add, srcs=[o_proj], dsts=[reduced_o_proj_out_memref], replica_groups=replica_groups, dtype=o_proj.dtype)

  if sp_rmsnorm_param and not INTERLEAVE_RS_AG: # SP RMSNorm, but don't interleave RS/AG.
    # All reduce-scatter are done in the loop nest above.
    # Now do all the SP-RMSNorm and all-gather here.
    [sp_rmsnorm_ag(t) for t in nl.static_range(num_tiles)]


def rmsnorm_mlp_block(mlp_X, mlp_X_b_offset, reduced_o_proj_out, Gamma_mlp, W_gate, W_up, W_down, Out,
                      replica_groups, num_tiles, sp_rmsnorm_param=None, eps=1e-6):
  """
  Perform MLP with tiling along seqlen dimension.
  """
  bs, seqlen, D = reduced_o_proj_out.shape
  i_b = nl.arange(bs)[:, None, None]
  i_x_D = nl.arange(D)[None, None, :]
  # Tile along the sequence length dimension.
  assert seqlen % num_tiles == 0, f'Sequence length of {seqlen} is not divisible by {num_tiles} tiles.'
  tile_size = seqlen // num_tiles

  # Preload gamma into sbuf and use it for all tiles.
  if sp_rmsnorm_param:
    sp_seqlen = seqlen // len(replica_groups[0])
    sp_tile_size = sp_seqlen // num_tiles
    rmsnorm_tile_size = 128 if (sp_tile_size // n_prgs) > 128 else (sp_tile_size // n_prgs)
    g_tensor_sbuf = nl.load(sp_rmsnorm_param.gamma)
    gamma_sbuf = g_tensor_sbuf.broadcast_to([rmsnorm_tile_size, D])

  def rs(t): # Reduce-scatter
    assert Out.shape[1] * len(replica_groups[0]) == mlp_out.shape[1], f'{Out.shape[1]} * {len(replica_groups[0])} == {mlp_out.shape[1]}, {reduced_o_proj_out.shape}'

    i_sl_t = nl.arange(tile_size)[None, :, None] + tile_size * t
    mlp_out_memref = mlp_out[i_b, i_sl_t, i_x_D]

    tiled_sp_num_rows = Out.shape[1] // num_tiles
    i_sl_t_sp = nl.arange(tiled_sp_num_rows)[None, :, None] + tiled_sp_num_rows * t
    Out_ref = Out[i_b, i_sl_t_sp, i_x_D]
    nccl.reduce_scatter(op=np.add, srcs=[mlp_out_memref], dsts=[Out_ref], replica_groups=replica_groups, reduce_scatter_dim=1, dtype=Out.dtype)

  def sp_rmsnorm_ag(t): # Sequence parallel RMSNorm, followed by all_gather.
    sp_tile_fused_add_rmsnorm_all_gather(
        Out, sp_rmsnorm_param.fused_add_sp, gamma_sbuf, sp_rmsnorm_param.ag_rmsnorm,
        t, tile_size, sp_tile_size, replica_groups)

  mlp_out = nl.ndarray(shape = [bs, seqlen, D], dtype = Out.dtype, buffer=nl.shared_hbm)

  # Further divide each tile into n_prgs shards.
  shard_size = tile_size // n_prgs
  assert shard_size * n_prgs == tile_size

  for t in nl.static_range(num_tiles):
    i_sl_t = nl.arange(shard_size)[None, :, None] + tile_size * t + shard_size * prg_id
    i_sl_lnc_t = nl.arange(tile_size)[None, :, None] + tile_size * t  # Need all NCs to finish before reduce CC.

    # Declaring mlp_out locally in the loop body is causing DRAM memspace not found error after DMA optimization.
    # mlp_out = nl.ndarray(shape = [bs, tile_size, D], dtype = Out.dtype, buffer=nl.shared_hbm)
    mlp_out_shard_ref = mlp_out[i_b, i_sl_t, i_x_D]

    reduced_o_proj_out_memref = reduced_o_proj_out[i_b, i_sl_t, i_x_D]
    if sp_rmsnorm_param is None: # Pure-TP, then fuse add and rmsnorm in MLP.
      X_memref = mlp_X[i_b + mlp_X_b_offset, i_sl_t, i_x_D]  # Return a memref tile.
      mlp_fused_add_kernel("MLP", X_memref, reduced_o_proj_out_memref, Gamma_mlp.as_tile(), W_gate.as_tile(), W_up.as_tile(), W_down.as_tile(), eps, mlp_out_shard_ref, norm_type=NormType.RMS_NORM, store_add=False)
    else:  # No fused add or RMSNorm.
      mlp_kernel("MLP", reduced_o_proj_out_memref, Gamma_mlp.as_tile(), W_gate.as_tile(), W_up.as_tile(), W_down.as_tile(), eps, mlp_out_shard_ref, norm_type=NormType.NO_NORM)

    # Don't need a core barrier on mlp_out_memref, because reduce acts as a barrier.
    mlp_out_memref = mlp_out[i_b, i_sl_lnc_t, i_x_D]

    if sp_rmsnorm_param: # SP RMSNorm
      rs(t)  # Do reduce_scatter
      if INTERLEAVE_RS_AG:
        sp_rmsnorm_ag(t) # If we interleave RS and AG, do SP-RMSNorm and all-gather right here.
    else: # Pure-TP, do an all-reduce here.
      Out_memref = Out[i_b, i_sl_lnc_t, i_x_D]
      nccl.all_reduce(op=np.add, srcs=[mlp_out_memref], dsts=[Out_memref], replica_groups=replica_groups, dtype = mlp_out.dtype)

  if sp_rmsnorm_param and not INTERLEAVE_RS_AG: # SP RMSNorm, but don't interleave RS/AG.
    # All reduce-scatter are done in the loop nest above.
    # Now do all the SP-RMSNorm and all-gather here.
    [sp_rmsnorm_ag(t) for t in nl.static_range(num_tiles)]


def llama3_transfomer_fwd_tp(X, W_qkv, W_o, W_gate, W_up, W_down, Gamma_qkv, Gamma_mlp, RoPE_cos, RoPE_sin, Out,
                             num_workers, num_tiles, attn_kernel_name, softmax_scale):
  """
  Transformer context encoding fwd pass with pure tensor parallel implementation.
  Current implementation is hard-coded to 4 layers.
  """
  __init_spmd_grid_size()

  bs, seqlen, D = X.shape

  # Hard code all the layer's data, and re-use weights.
  qkv_proj_out_0 = nl.ndarray(shape = [X.shape[0], X.shape[1], W_qkv.shape[1]], dtype = X.dtype, buffer=nl.shared_hbm)
  qkv_proj_out_1 = nl.ndarray(shape = [X.shape[0], X.shape[1], W_qkv.shape[1]], dtype = X.dtype, buffer=nl.shared_hbm)
  qkv_proj_out_2 = nl.ndarray(shape = [X.shape[0], X.shape[1], W_qkv.shape[1]], dtype = X.dtype, buffer=nl.shared_hbm)
  qkv_proj_out_3 = nl.ndarray(shape = [X.shape[0], X.shape[1], W_qkv.shape[1]], dtype = X.dtype, buffer=nl.shared_hbm)

  reduced_o_proj_out_0 = nl.ndarray(shape = X.shape, dtype = X.dtype, buffer=nl.shared_hbm)
  reduced_o_proj_out_1 = nl.ndarray(shape = X.shape, dtype = X.dtype, buffer=nl.shared_hbm)
  reduced_o_proj_out_2 = nl.ndarray(shape = X.shape, dtype = X.dtype, buffer=nl.shared_hbm)
  reduced_o_proj_out_3 = nl.ndarray(shape = X.shape, dtype = X.dtype, buffer=nl.shared_hbm)

  mlp_out_0 = nl.ndarray(shape = X.shape, dtype = X.dtype, buffer=nl.shared_hbm)
  mlp_out_1 = nl.ndarray(shape = X.shape, dtype = X.dtype, buffer=nl.shared_hbm)
  mlp_out_2 = nl.ndarray(shape = X.shape, dtype = X.dtype, buffer=nl.shared_hbm)

  replica_groups = [list(range(num_workers))]

  # First layer
  tiled_rmsnorm_qkv(X, W_qkv, Gamma_qkv, qkv_proj_out_0, num_tiles, norm_type=NormType.RMS_NORM)
  attention_output_proj(qkv_proj_out_0, RoPE_cos, RoPE_sin, W_o, reduced_o_proj_out_0, attn_kernel_name, softmax_scale, replica_groups, num_tiles)
  rmsnorm_mlp_block(X, 0, reduced_o_proj_out_0, Gamma_mlp, W_gate, W_up, W_down, mlp_out_0, replica_groups, num_tiles)

  # Second layer.
  # Note that after rmsnorm_qkv_isa_fused_add_kernel, mlp_out_0 has the fused add result.
  tiled_rmsnorm_qkv(mlp_out_0, W_qkv, Gamma_qkv, qkv_proj_out_1, num_tiles, norm_type=NormType.RMS_NORM, fused_add_op0=X, fused_add_op1=reduced_o_proj_out_0)
  attention_output_proj(qkv_proj_out_1, RoPE_cos, RoPE_sin, W_o, reduced_o_proj_out_1, attn_kernel_name, softmax_scale, replica_groups, num_tiles)
  rmsnorm_mlp_block(mlp_out_0, 0, reduced_o_proj_out_1, Gamma_mlp, W_gate, W_up, W_down, mlp_out_1, replica_groups, num_tiles)

  tiled_rmsnorm_qkv(mlp_out_1, W_qkv, Gamma_qkv, qkv_proj_out_2, num_tiles, norm_type=NormType.RMS_NORM, fused_add_op0=mlp_out_0, fused_add_op1=reduced_o_proj_out_1)
  attention_output_proj(qkv_proj_out_2, RoPE_cos, RoPE_sin, W_o, reduced_o_proj_out_2, attn_kernel_name, softmax_scale, replica_groups, num_tiles)
  rmsnorm_mlp_block(mlp_out_1, 0, reduced_o_proj_out_2, Gamma_mlp, W_gate, W_up, W_down, mlp_out_2, replica_groups, num_tiles)

  tiled_rmsnorm_qkv(mlp_out_2, W_qkv, Gamma_qkv, qkv_proj_out_3, num_tiles, norm_type=NormType.RMS_NORM, fused_add_op0=mlp_out_1, fused_add_op1=reduced_o_proj_out_2)
  attention_output_proj(qkv_proj_out_3, RoPE_cos, RoPE_sin, W_o, reduced_o_proj_out_3, attn_kernel_name, softmax_scale, replica_groups, num_tiles)
  rmsnorm_mlp_block(mlp_out_2, 0, reduced_o_proj_out_3, Gamma_mlp, W_gate, W_up, W_down, Out, replica_groups, num_tiles)


def llama3_transfomer_fwd_rmsnorm_sp(X, W_qkv, W_o, W_gate, W_up, W_down, Gamma_qkv, Gamma_mlp, RoPE_cos, RoPE_sin, Out,
                                     num_workers, num_tiles, attn_kernel_name, softmax_scale):
  """
  Transformer context encoding fwd pass with sequence parallel RMSNorm and tensor parallel for attention block and MLP..
  Current implementation is hard-coded to 4 layers.
  """
  __init_spmd_grid_size()

  bs, seqlen, D = X.shape
  sp_seqlen = seqlen // num_workers
  assert sp_seqlen * num_workers == seqlen, f'seqlen {seqlen} is not divisible by num_workers {num_workers}'

  def fused_add_rmsnorm(a_tensor, b_tensor, g_tensor, out_tensor, num_tiles, replica_groups):
    """
    a_tensor, b_tensor, and out_tensor should all be in [bs, sp_seqlen, D] shape.
    if b_tensor is provided, fuse add will be performed and the summation result will be stored into b_tensor.
    """
    assert a_tensor.shape == (bs, sp_seqlen, D)
    assert b_tensor == None or b_tensor.shape == (bs, sp_seqlen, D)
    assert tuple(out_tensor.shape) == (bs, seqlen, D)

    # Tiling along the seqlen dimension by num_tiles.
    # For each of the tile within the sp_seqlen dimension, we further divide
    # into num_rmsnorm_tiles, each with 128 rows to match partition size in
    # SBUF.
    tile_size = seqlen // num_tiles # Tile size in the full tensor.
    sp_tile_size = sp_seqlen // num_tiles
    rmsnorm_tile_size = 128 if (sp_tile_size // n_prgs) > 128 else (sp_tile_size // n_prgs)
    num_rmsnorm_tiles = sp_tile_size // rmsnorm_tile_size

    assert rmsnorm_tile_size * num_rmsnorm_tiles == sp_tile_size

    i_D = nl.arange(D)[None, :]

    g_tensor_sbuf = nl.load(g_tensor)
    gamma_sbuf = g_tensor_sbuf.broadcast_to([rmsnorm_tile_size, D])

    dtype = a_tensor.dtype
    assert dtype == out_tensor.dtype

    for t in nl.static_range(num_tiles):
      sp_tile_fused_add_rmsnorm_all_gather(a_tensor, b_tensor, gamma_sbuf, out_tensor, t, tile_size, sp_tile_size, replica_groups)

  replica_groups = [list(range(num_workers))]

  # Hard code all the layer's data, and re-use weights.
  qkv_proj_out_0 = nl.ndarray(shape = [X.shape[0], X.shape[1], W_qkv.shape[1]], dtype = X.dtype, buffer=nl.shared_hbm)
  qkv_proj_out_1 = nl.ndarray(shape = [X.shape[0], X.shape[1], W_qkv.shape[1]], dtype = X.dtype, buffer=nl.shared_hbm)
  qkv_proj_out_2 = nl.ndarray(shape = [X.shape[0], X.shape[1], W_qkv.shape[1]], dtype = X.dtype, buffer=nl.shared_hbm)
  qkv_proj_out_3 = nl.ndarray(shape = [X.shape[0], X.shape[1], W_qkv.shape[1]], dtype = X.dtype, buffer=nl.shared_hbm)

  X_sp_shape = [bs, seqlen // num_workers, D]  # Sharded along seqlen dimension.
  assert X_sp_shape[1] * num_workers == seqlen, f'seqlen {seqlen} is not divisible by num_workers {num_workers}'

  reduced_o_proj_out_0 = nl.ndarray(shape = X_sp_shape, dtype = X.dtype, buffer=nl.shared_hbm)
  reduced_o_proj_out_1 = nl.ndarray(shape = X_sp_shape, dtype = X.dtype, buffer=nl.shared_hbm)
  reduced_o_proj_out_2 = nl.ndarray(shape = X_sp_shape, dtype = X.dtype, buffer=nl.shared_hbm)
  reduced_o_proj_out_3 = nl.ndarray(shape = X_sp_shape, dtype = X.dtype, buffer=nl.shared_hbm)

  mlp_out_0 = nl.ndarray(shape = X_sp_shape, dtype = X.dtype, buffer=nl.shared_hbm)
  mlp_out_1 = nl.ndarray(shape = X_sp_shape, dtype = X.dtype, buffer=nl.shared_hbm)
  mlp_out_2 = nl.ndarray(shape = X_sp_shape, dtype = X.dtype, buffer=nl.shared_hbm)
  mlp_out_3 = nl.ndarray(shape = X_sp_shape, dtype = X.dtype, buffer=nl.shared_hbm)


  # 'Collective instruction cannot read IO tensors'.
  # Normally this would be taken care of by CoalesceCCOp, which automatically
  # adds extra local tensors and does the copies (just like what we do here).
  # But unfortunately when the pass does the extra copy for us, it is done incorrectly.
  # So let's do it explicitly here so CoalesceCCOp won't do the wrong transformation.
  X_local = nl.ndarray(shape = X.shape, dtype = X.dtype, buffer=nl.shared_hbm)
  Out_local = nl.ndarray(shape = Out.shape, dtype = X.dtype, buffer=nl.shared_hbm)

  nisa._tiled_offloaded_memcpy(src=      X[nl.arange(bs)[:, None, None], nl.arange(seqlen // n_prgs)[None, :, None] + (seqlen // n_prgs) * prg_id, nl.arange(D)[None, None, :]],
                               dst=X_local[nl.arange(bs)[:, None, None], nl.arange(seqlen // n_prgs)[None, :, None] + (seqlen // n_prgs) * prg_id, nl.arange(D)[None, None, :]])

  X_sp = nl.ndarray(shape = [bs, sp_seqlen, D], dtype = X.dtype, buffer=nl.shared_hbm)
  fused_add_sp = nl.ndarray(shape = [bs, sp_seqlen, D], dtype = X.dtype, buffer=nl.shared_hbm)
  full_rmsnorm_result = nl.ndarray(shape = X.shape, dtype = X.dtype, buffer=nl.shared_hbm)

  # First layer.
  # Start off with a reduce scatter that is actually dummy (essentially slicing input), and save it in
  # fused_add_sp because this tensor will be used in residual add.
  nccl.reduce_scatter(op=np.add, srcs=[X_local], dsts=[fused_add_sp], replica_groups=replica_groups, reduce_scatter_dim=1, dtype=X.dtype)
  fused_add_rmsnorm(fused_add_sp, None, Gamma_qkv, full_rmsnorm_result, num_tiles, replica_groups)

  # Gamma_qkv should not be used when not rmsnorm.
  tiled_rmsnorm_qkv(full_rmsnorm_result, W_qkv, Gamma_qkv, qkv_proj_out_0, num_tiles, norm_type=NormType.NO_NORM)
  attention_output_proj(qkv_proj_out_0, RoPE_cos, RoPE_sin, W_o, reduced_o_proj_out_0, attn_kernel_name, softmax_scale, replica_groups, num_tiles,
                        sp_rmsnorm_param=SpRmsNormParam(fused_add_sp, Gamma_mlp, full_rmsnorm_result))
  rmsnorm_mlp_block(None, 0, full_rmsnorm_result, Gamma_mlp, W_gate, W_up, W_down, mlp_out_0, replica_groups, num_tiles,
                    sp_rmsnorm_param=SpRmsNormParam(fused_add_sp, Gamma_qkv, full_rmsnorm_result))

  # Second layer.
  tiled_rmsnorm_qkv(full_rmsnorm_result, W_qkv, Gamma_qkv, qkv_proj_out_1, num_tiles, norm_type=NormType.NO_NORM)
  attention_output_proj(qkv_proj_out_1, RoPE_cos, RoPE_sin, W_o, reduced_o_proj_out_1, attn_kernel_name, softmax_scale, replica_groups, num_tiles,
                        sp_rmsnorm_param=SpRmsNormParam(fused_add_sp, Gamma_mlp, full_rmsnorm_result))
  rmsnorm_mlp_block(None, 0, full_rmsnorm_result, Gamma_mlp, W_gate, W_up, W_down, mlp_out_1, replica_groups, num_tiles,
                    sp_rmsnorm_param=SpRmsNormParam(fused_add_sp, Gamma_qkv, full_rmsnorm_result))

  # Third layer.
  tiled_rmsnorm_qkv(full_rmsnorm_result, W_qkv, Gamma_qkv, qkv_proj_out_2, num_tiles, norm_type=NormType.NO_NORM)
  attention_output_proj(qkv_proj_out_2, RoPE_cos, RoPE_sin, W_o, reduced_o_proj_out_2, attn_kernel_name, softmax_scale, replica_groups, num_tiles,
                        sp_rmsnorm_param=SpRmsNormParam(fused_add_sp, Gamma_mlp, full_rmsnorm_result))
  rmsnorm_mlp_block(None, 0, full_rmsnorm_result, Gamma_mlp, W_gate, W_up, W_down, mlp_out_2, replica_groups, num_tiles,
                    sp_rmsnorm_param=SpRmsNormParam(fused_add_sp, Gamma_qkv, full_rmsnorm_result))

  # Fourth layer.
  tiled_rmsnorm_qkv(full_rmsnorm_result, W_qkv, Gamma_qkv, qkv_proj_out_3, num_tiles, norm_type=NormType.NO_NORM)
  attention_output_proj(qkv_proj_out_3, RoPE_cos, RoPE_sin, W_o, reduced_o_proj_out_3, attn_kernel_name, softmax_scale, replica_groups, num_tiles,
                        sp_rmsnorm_param=SpRmsNormParam(fused_add_sp, Gamma_mlp, full_rmsnorm_result))
  rmsnorm_mlp_block(None, 0, full_rmsnorm_result, Gamma_mlp, W_gate, W_up, W_down, mlp_out_3, replica_groups, num_tiles,
                    sp_rmsnorm_param=SpRmsNormParam(fused_add_sp, Gamma_qkv, Out_local))

  nisa._tiled_offloaded_memcpy(src=Out_local[nl.arange(bs)[:, None, None], nl.arange(seqlen // n_prgs)[None, :, None] + (seqlen // n_prgs) * prg_id, nl.arange(D)[None, None, :]],
                               dst=      Out[nl.arange(bs)[:, None, None], nl.arange(seqlen // n_prgs)[None, :, None] + (seqlen // n_prgs) * prg_id, nl.arange(D)[None, None, :]])
