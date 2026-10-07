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
  input (x) is row-sharded:    each rank holds (M_LOCAL, K)
  weight (w) is column-sharded: each rank holds (K, N_LOCAL)
  full output: (RANK_N * M_LOCAL, RANK_N * N_LOCAL)

Ring algorithm (per rank):
  buf_cur = [local input shard]
  for step in 0 .. RANK_N-1:
    src_rank = processing_rank_id(step)        # whose data is in buf_cur
    result[src_rank] = buf_cur @ weight           # local matmul
    if step < RANK_N - 1:
      CPI(buf_cur -> buf_next)                 # start next step's transfer
    swap(buf_cur, buf_next)

The ``if step < RANK_N - 1`` guard matters: we don't need the final CPI.

This kernel runs on LNC=1 (one physical core per rank). The per-rank
shape (M_LOCAL × N_LOCAL) is fixed to the hardware tile-size limits.
CPI requires a valid hardware ring; not every world-size × LNC combination
forms one — world_size=8 with LNC=1 (one device, 8 cores) is the canonical
config for this sample.
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl


RANK_N  = 8    # ring length (= world size)
K       = 2048 # shared contraction dim
K_TILES = K // nl.tile_size.pmax


# NKI_EXAMPLE_AGMM_RING_BEGIN
@nki.jit
def allgather_matmul_ring(input_shard, weight_shard, replica_group):
  """Ring all-gather + matmul.

  Args:
      input_shard (nl.ndarray): [M_LOCAL, K] — this rank's row-slice of the
          input activation. **Rotating**: each ring step this shard is passed
          to the next rank via CPI so every rank eventually sees all slices.
      weight_shard (nl.ndarray): [K, N_LOCAL] — this rank's column-slice of
          the weight matrix. **Stationary**: never sent over the ring; each
          rank always multiplies against its own local weight shard.
      replica_group (ncc.ReplicaGroup): Ring of all ranks.

  Note:
      This sample illustrates the ring algorithm and compute/communication
      overlap on a single trn2 device (LNC=1, 8 cores). Each rank's M_LOCAL (= nl.tile_size.pmax) fills the Tensor Engine
      partition dim, so each rank fully utilizes its core. The kernel uses
      ``nl.hbm`` (private per-core HBM) for the ring buffers because each
      rank owns exactly one physical core and no cross-core sharing is needed.
      For production-grade performance with LNC=2 and shared-HBM ring buffers,
      see the fine-grained collective communication API:
      https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/library/api/fgcc.html

  Returns:
      nl.ndarray of shape [RANK_N, nl.tile_size.pmax, nl.tile_size.gemm_moving_fmax].
      Slot ``r`` contains the matmul contribution from rank ``r``'s input shard.
  """
  # Local aliases for hardware tile-size constants — improves readability
  # without polluting the module namespace.
  M_LOCAL = nl.tile_size.pmax              # partition dim per rank (= K_TILE)
  N_LOCAL = nl.tile_size.gemm_moving_fmax  # free dim per rank
  K_TILE  = nl.tile_size.pmax              # contract-dim tile

  # Two ping-pong ring buffers in private per-core HBM (nl.hbm). We alternate
  # between them across steps: while one buffer is being sent to the next rank
  # via CPI, the matmul reads from the same buffer concurrently (independent
  # readers), and the received data lands in the OTHER buffer so there is no
  # write-after-read hazard between consecutive steps. Collectives cannot
  # source from IO tensors directly, so we seed buf0 with a DMA copy below.
  buf0 = nl.ndarray((M_LOCAL, K), dtype=input_shard.dtype, buffer=nl.hbm,
                   name="ring_buf0")
  buf1 = nl.ndarray((M_LOCAL, K), dtype=input_shard.dtype, buffer=nl.hbm,
                   name="ring_buf1")

  # Output: one (M_LOCAL, N_LOCAL) slot per source-rank.
  # NOTE: nl.shared_hbm is a temporary workaround — the compiler currently
  # disallows nl.hbm for kernel return tensors but still lowers the
  # allocation to private HBM. Change to nl.hbm once the restriction lifts.
  out = nl.ndarray((RANK_N, M_LOCAL, N_LOCAL), dtype=input_shard.dtype,
                  buffer=nl.shared_hbm, name="out")

  # Seed the ring: copy local input_shard into buf0.
  nisa.dma_copy(dst=buf0, src=input_shard)

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

    # Local matmul (M_LOCAL, N_LOCAL) = input_shard_cur @ weight_shard.
    # Tiled over K (K_TILES × K_TILE accumulated into PSUM).
    # N_LOCAL fits in a single Tensor Engine free-dim tile, so no N-loop.
    # M fits in a single Tensor Engine partition.
    result_sbuf = nl.ndarray((M_LOCAL, N_LOCAL), dtype=input_shard.dtype,
                             buffer=nl.sbuf)
    psum = nl.zeros((M_LOCAL, N_LOCAL), dtype=nl.float32, buffer=nl.psum)
    for kt in nl.affine_range(K_TILES):
      k_slice = slice(kt * K_TILE, (kt + 1) * K_TILE)
      # Allocate fresh SBUF tiles each K-step instead of reusing the same
      # buffer. Reusing would create a write-after-write dependency that
      # forces the K-tile DMAs to execute serially; fresh tiles let the
      # compiler pipeline the next K-tile's load against the current
      # nc_matmul.
      input_tile = nl.ndarray((K_TILE, M_LOCAL), dtype=input_shard.dtype,
                              buffer=nl.sbuf)
      weight_tile = nl.ndarray((K_TILE, N_LOCAL), dtype=weight_shard.dtype,
                               buffer=nl.sbuf)
      # nc_matmul wants stationary in [K, M] layout. buf_cur is
      # [M, K], so dma_transpose swaps axes during the load.
      nisa.dma_transpose(dst=input_tile, src=buf_cur[:, k_slice])
      nisa.dma_copy(dst=weight_tile, src=weight_shard[k_slice, :])
      # nc_matmul accumulates into psum (pre-zeroed above), so each
      # K-tile adds its partial product to the running sum.
      nisa.nc_matmul(dst=psum, stationary=input_tile, moving=weight_tile)
    nisa.tensor_copy(dst=result_sbuf, src=psum)

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
