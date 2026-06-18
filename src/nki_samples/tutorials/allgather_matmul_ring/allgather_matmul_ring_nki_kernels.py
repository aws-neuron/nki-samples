"""
Copyright (C) 2026, Amazon.com. All Rights Reserved

NKI implementation for the all-gather + matmul ring tutorial.

This kernel performs a fused all-gather + matmul along a ring of ranks,
using ``nki.collectives.collective_permute_implicit`` (CPI) to overlap
communication with compute. Each step, a rank computes a local matmul
using the LHS fragment currently in its ring buffer, then passes the
fragment on to the next rank in the ring while receiving the previous
rank's fragment — the scheduler places the matmul of step (i) and the
CPI of step (i) on disjoint engines, so they run concurrently.

The matmul is row-parallel (LHS row-sharded, RHS column-sharded). After
RANK_N ring steps, every rank has computed one (M_LOCAL, N_LOCAL) slot of
the fully-gathered output for every source rank.

Shape convention:
  lhs (x) is row-sharded:    each rank holds (M_LOCAL, K)
  rhs (w) is column-sharded: each rank holds (K, N_LOCAL)
  full output: (M_full, N_full) = (M_LOCAL * RANK_N, N_LOCAL * RANK_N)

Ring algorithm (per rank):
  buf_cur = [local lhs shard]
  for step in 0 .. RANK_N-1:
    src_rank = processing_rank_id(step)        # whose data is in buf_cur
    result[src_rank] = buf_cur @ rhs           # local matmul
    if step < RANK_N - 1:
      CPI(buf_cur -> buf_next)                 # start next step's transfer
    swap(buf_cur, buf_next)

The ``if step < RANK_N - 1`` guard matters: we don't need the final CPI.

This kernel is sized for LNC=1: M_LOCAL=128 matches the Tensor Engine
partition-dim limit of a single physical core, so each rank fully utilizes
its core. CPI requires a valid hardware ring; not every world-size × LNC
combination forms one — world_size=8 with LNC=1 (one device, 8 cores) is
the canonical config for this sample.
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl


# Problem shape and tiling. Fixed at module level so the kernel signature
# only carries the per-rank tensors and the replica group.
RANK_N = 8           # world size, also ring length
M_LOCAL = 128        # per-rank M slice (= TE partition-dim limit)
N_LOCAL = 512        # per-rank N slice (RHS column shard width)
K = 2048             # shared contraction dim
K_TILE = 128         # contract-dim tile (= TE partition-dim limit)
N_TILE = 512         # free-dim tile (<= TE free-dim limit)
K_TILES = K // K_TILE
N_TILES = N_LOCAL // N_TILE


# NKI_EXAMPLE_AGMM_RING_BEGIN
@nki.jit
def allgather_matmul_ring(lhs_shard, rhs_shard, replica_group):
  """Ring all-gather + matmul.

  Args:
      lhs_shard (nl.ndarray): [M_LOCAL, K] — this rank's LHS row-slice.
      rhs_shard (nl.ndarray): [K, N_LOCAL] — this rank's RHS column-slice.
      replica_group (ncc.ReplicaGroup): Ring of all ranks.

  Returns:
      nl.ndarray of shape [RANK_N, M_LOCAL, N_LOCAL]. Slot ``r`` contains the
      matmul contribution from rank ``r``'s LHS shard (lhs_shard_r @ rhs_shard).
  """

  # Two ping-pong ring buffers on per-core HBM. Collectives cannot source
  # from IO tensors directly, so we also seed with a DMA copy below.
  buf0 = nl.ndarray((M_LOCAL, K), dtype=lhs_shard.dtype, buffer=nl.hbm,
                   name="ring_buf0")
  buf1 = nl.ndarray((M_LOCAL, K), dtype=lhs_shard.dtype, buffer=nl.hbm,
                   name="ring_buf1")

  # Output: one (M_LOCAL, N_LOCAL) slot per source-rank.
  out = nl.ndarray((RANK_N, M_LOCAL, N_LOCAL), dtype=lhs_shard.dtype,
                  buffer=nl.shared_hbm, name="out")

  # Seed the ring: copy local lhs_shard into buf0.
  nisa.dma_copy(dst=buf0, src=lhs_shard)

  # `step` is an ordinary Python integer here (meta-programming): the
  # compiler specializes the kernel body for each concrete value of step.
  for step in range(RANK_N):
    # Alternate the two ring buffers across steps.
    if step % 2 == 0:
      buf_cur, buf_next = buf0, buf1
    else:
      buf_cur, buf_next = buf1, buf0

    # src_rank is a runtime-valued rank ID held in a register. It can only
    # be used as scalar_offset in an access pattern (not compared, not
    # materialized as a Python value).
    src_rank = ncc.collective_permute_implicit_current_processing_rank_id(
        iteration_id=step, replica_group=replica_group,
    )

    # Launch the CPI for the NEXT step first. The matmul below and this
    # CPI both read buf_cur — independent readers — and CPI writes to a
    # buffer not touched until the next iteration. Placing the CPI first
    # hints the scheduler to overlap the two on different engines.
    if step < RANK_N - 1:
      ncc.collective_permute_implicit(
          srcs_by_channel=[[buf_cur]],
          dsts_by_channel=[[buf_next]],
          replica_group=replica_group,
      )

    # Local matmul (M_LOCAL, N_LOCAL) = lhs_shard_cur @ rhs_shard.
    # Tiled over K (K_TILES × K_TILE accumulated into PSUM) and N
    # (N_TILES × N_TILE). M fits in a single Tensor Engine partition.
    result_sbuf = nl.ndarray((M_LOCAL, N_LOCAL), dtype=lhs_shard.dtype,
                             buffer=nl.sbuf)
    for nt in nl.affine_range(N_TILES):
      n0 = nt * N_TILE
      psum = nl.zeros((M_LOCAL, N_TILE), dtype=nl.float32, buffer=nl.psum)
      for kt in nl.affine_range(K_TILES):
        k0 = kt * K_TILE
        # Allocate fresh SBUF tiles each K-step instead of reusing the same
        # buffer. Reusing would create a write-after-write dependency that
        # forces the K-tile DMAs to execute serially; fresh tiles let the
        # compiler pipeline the next K-tile's load against the current
        # nc_matmul.
        lhs_tile = nl.ndarray((K_TILE, M_LOCAL), dtype=lhs_shard.dtype,
                              buffer=nl.sbuf)
        rhs_tile = nl.ndarray((K_TILE, N_TILE), dtype=rhs_shard.dtype,
                              buffer=nl.sbuf)
        # nc_matmul wants stationary in [K, M] layout. buf_cur is
        # [M, K], so dma_transpose swaps axes during the load.
        nisa.dma_transpose(dst=lhs_tile, src=buf_cur[:, k0 : k0 + K_TILE])
        nisa.dma_copy(dst=rhs_tile,
                      src=rhs_shard[k0 : k0 + K_TILE, n0 : n0 + N_TILE])
        # nc_matmul accumulates into psum (pre-zeroed above), so each
        # K-tile adds its partial product to the running sum.
        nisa.nc_matmul(dst=psum, stationary=lhs_tile, moving=rhs_tile)
      nisa.tensor_copy(dst=result_sbuf[:, n0 : n0 + N_TILE], src=psum)

    # Write this step's result into the src_rank slot of `out`.
    # `pattern=[[N_LOCAL, M_LOCAL], [1, N_LOCAL]]` describes the inner
    # (M, N) tile in row-major order; `scalar_offset=src_rank` with
    # `indirect_dim=0` selects which of the leading RANK_N output slots
    # to write at runtime (src_rank is a register-valued rank id).
    nisa.dma_copy(
        dst=out.ap(
            pattern=[[N_LOCAL, M_LOCAL], [1, N_LOCAL]],
            offset=0,
            scalar_offset=src_rank,
            indirect_dim=0,
        ),
        src=result_sbuf,
    )

  return out
# NKI_EXAMPLE_AGMM_RING_END
