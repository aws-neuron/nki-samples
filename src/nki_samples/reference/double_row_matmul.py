"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

kernels - Builtin high performance NKI kernels.

"""

from neuronxcc import nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl

@nki.jit
def quantized_double_row_matmul(
    lhs,
    rhs_quantized, rhs_scale,
    # Meta-parameters
    TILES_IN_BLOCK_M,
    TILES_IN_BLOCK_N,
    TILES_IN_BLOCK_K
):
  """NKI kernel to compute a matrix multiplication by blocking along all dimensions
     and performing fp8_e4m3 quantization on lhs matrix.
  
  Args:
      lhs: an unquantized input tensor of shape [M,K], where K is a multiple of 128 *
        TILES_IN_BLOCK_K and M is a multiple of 128 * TILES_IN_BLOCK_M.  It is the
        left-hand-side argument of the matrix multiplication.
      rhs_quantized: a pre-quantized input tensor of dtype float8_e4m3 and of shape 
        [K // 2,2 * N] (reshaped from the original [K,N] rhs) where K is a multiple of 128 *
        TILES_IN_BLOCK_K and N is a multiple of 512 * TILES_IN_BLOCK_N. It is the
        right-hand-side argument of the matrix multiplication. See test_double_row_matmul.py
        for the expected reshape to be performed on the original rhs matrix.
      rhs_scale: the quantization column-wise scale of rhs of shape [128, N] that is 
        pre-broadcasted from [1, N].
      TILES_IN_BLOCK_*: meta parameters to control blocking dimensions
  Returns:
      result: the resulting output tensor of shape [M,N]
  """

  assert rhs_quantized.dtype == nl.float8_e4m3, "rhs must be pre-quantized to dtype float8_e4m3"

  M, K = lhs.shape
  K_RESHAPED, N_RESHAPED = rhs_quantized.shape
  K_ = 2 * K_RESHAPED

  assert K == K_, "lhs and rhs must have the same contraction dimension"

  assert N_RESHAPED % 2 == 0, f"N_RESHAPED={N_RESHAPED} must be divisible by 2"
  N = N_RESHAPED // 2

  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  TILE_K = nl.tile_size.pmax  # 128
  TILE_N = nl.tile_size.gemm_moving_fmax  # 512

  BLOCK_M = TILE_M * TILES_IN_BLOCK_M
  BLOCK_N = TILE_N * TILES_IN_BLOCK_N
  BLOCK_K = TILE_K * TILES_IN_BLOCK_K

  assert M % BLOCK_M == 0
  assert N % BLOCK_N == 0
  assert K % BLOCK_K == 0

  # The size has to be multiple of block size.
  NUM_BLOCK_M = M // BLOCK_M
  NUM_BLOCK_N = N // BLOCK_N
  NUM_BLOCK_K = K // BLOCK_K

  # dtype fp8_e4m3 can represent [-240, 240].
  FP8_RANGE = 240

  assert TILES_IN_BLOCK_K % 2 == 0, f"TILES_IN_BLOCK_K={TILES_IN_BLOCK_K} must be even to load 2 tiles at a time for double row matmul"

  result = nl.ndarray((M, N), dtype=lhs.dtype, buffer=nl.shared_hbm)

  # Blocking M dimension (lhs partition dimension).
  for m in nl.affine_range(NUM_BLOCK_M):
    result_tiles = nl.zeros((TILE_M, NUM_BLOCK_N * TILES_IN_BLOCK_M * TILES_IN_BLOCK_N * TILE_N),
                            dtype=lhs.dtype,
                            buffer=nl.sbuf)

    # Blocking K dimension (the contraction dimension).
    # Use `sequential_range` because we do not want the compiler to change this loop by, 
    # for example, vectorizing it.
    for k in nl.sequential_range(NUM_BLOCK_K):
      lhsT_quantized_tiles = nl.ndarray((TILES_IN_BLOCK_M, nl.par_dim(TILE_M), BLOCK_K),
                                        dtype=nl.float8_e4m3,
                                        buffer=nl.sbuf)
      lhsT_scale_tiles = nl.ndarray((TILES_IN_BLOCK_M, nl.par_dim(TILE_M), 1),
                                    dtype=lhs.dtype,
                                    buffer=nl.sbuf)

      i_lhs = nl.mgrid[0:TILE_M, 0:BLOCK_K]
      for bm_l in nl.affine_range(TILES_IN_BLOCK_M):
        # Load and quantize tiles from rhs,
        # setting the load tile to [TILE_M, BLOCK_K] to optimize DMA performance.
        lhs_i_m = m * BLOCK_M + bm_l * TILE_M + i_lhs.p
        lhs_i_k = k * BLOCK_K + i_lhs.x

        tile_block = nl.load(lhs[lhs_i_m, lhs_i_k])

        # FIXME: use nisa.tensor_scalar_reduce to fuse nl.abs and nisa.tensor_reduce into 
        #   1 operation.
        abs_tile_block = nl.abs(tile_block)
        lhsT_scale_tiles[bm_l] = nisa.tensor_reduce(nl.max,
                                                    abs_tile_block,
                                                    axis=[1])
        lhsT_scale_tiles[bm_l] = nl.divide(lhsT_scale_tiles[bm_l], FP8_RANGE)
        lhsT_quantized_tiles[bm_l] = nl.divide(tile_block, lhsT_scale_tiles[bm_l])

        # For each [TILE_M, TILE_K] tiles, since TILE_K == TILE_M and the K dimension needs to be
        # along the partition dimension, transpose said tiles in-place.
        for bk_l in nl.affine_range(TILES_IN_BLOCK_K):
          # FIXME: use dma_transpose instead of nc_transpose.
          lhsT_quantized_tiles[bm_l, :,
                               TILE_M * bk_l:(bk_l + 1) * TILE_M] = nisa.nc_transpose(lhsT_quantized_tiles[bm_l, :,
                                                                                                           TILE_M * bk_l:(bk_l + 1) * TILE_M])

      # Each lhs block's matmul results needs to be dequantized independent of another lhs block's matmul results.
      # scoped_result_tiles stores the non-dequantized matmul results scoped to each `for m` and `for k` loops.
      scoped_result_tiles = nl.zeros((TILE_M, NUM_BLOCK_N * TILES_IN_BLOCK_M * TILES_IN_BLOCK_N * TILE_N),
                                     dtype=lhs.dtype,
                                     buffer=nl.sbuf)

      for n in nl.affine_range(NUM_BLOCK_N):
        # Loading tiles from rhs,
        # setting the load tile to [TILE_K, 2 * BLOCK_N] to optimize DMA performance
        # (i.e. loading 2 rows of a rhs block at a time).
        i_rhs = nl.mgrid[0:TILE_K, 0:2 * BLOCK_N]

        rhs_quantized_tiles = nl.ndarray((TILES_IN_BLOCK_K // 2, nl.par_dim(TILE_K), 2 * BLOCK_N), dtype=rhs_quantized.dtype)
        for bk_r in nl.affine_range(TILES_IN_BLOCK_K // 2):
          rhs_quantized_i_k = (k * TILES_IN_BLOCK_K // 2 + bk_r) * TILE_K + i_rhs.p
          rhs_quantized_i_n = 2 * n * BLOCK_N + i_rhs.x
          rhs_quantized_tiles[bk_r] = nl.load(rhs_quantized[rhs_quantized_i_k, rhs_quantized_i_n])

        # Do matmul with all tiles in the loaded lhs and rhs blocks.
        i_res_mm = nl.mgrid[0:TILE_M, 0:TILE_N]
        for bm in nl.affine_range(TILES_IN_BLOCK_M):
          for bn in nl.affine_range(TILES_IN_BLOCK_N):
            res_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.bfloat16, buffer=nl.psum)
            for bk in nl.affine_range(TILES_IN_BLOCK_K // 2):
              i_k, i_tile_m, i_m = nl.mgrid[0:TILE_K, 0:2, 0:TILE_M]
              lhsT_double_tile = lhsT_quantized_tiles[
                bm,
                i_k,
                bk * (2 * TILE_M) + i_tile_m * TILE_M + i_m
              ]
              assert lhsT_double_tile.shape == (TILE_K, 2, TILE_M)

              i_k, i_tile_n, i_n = nl.mgrid[0:TILE_K, 0:2, 0:TILE_N]
              rhs_double_tile = rhs_quantized_tiles[
                bk,
                i_k,
                2 * bn * TILE_N + i_tile_n * TILE_N + i_n
              ]
              assert rhs_double_tile.shape == (TILE_K, 2, TILE_N)

              res_tile[...] += nisa.nc_matmul(lhsT_double_tile,
                                              rhs_double_tile,
                                              perf_mode='double_row_gen3')

            i_scoped_result_tiles_k = i_res_mm.p
            i_scoped_result_tiles_n = bm * (NUM_BLOCK_N * BLOCK_N) + n * BLOCK_N + bn * TILE_N + i_res_mm.x
            scoped_result_tiles[i_scoped_result_tiles_k, i_scoped_result_tiles_n] += res_tile[...]

      # FIXME: dequantize using both lhs and rhs scales using nisa.scalar_tensor_tensor when
      #   accumulating from PSUM to SBUF.
      # Partially dequantize matmul results using lhs block scale.
      i_scoped_result_tiles = nl.mgrid[0:TILE_K, 0:NUM_BLOCK_N * BLOCK_N]
      for bm in nl.affine_range(TILES_IN_BLOCK_M):
        result_tiles_i_k = i_scoped_result_tiles.p
        result_tiles_i_n = bm * NUM_BLOCK_N * BLOCK_N + i_scoped_result_tiles.x
        dequantized_tile_block = nisa.tensor_tensor(
          scoped_result_tiles[result_tiles_i_k, result_tiles_i_n],
          lhsT_scale_tiles[bm],
          nl.multiply
        )

        result_tiles[result_tiles_i_k, result_tiles_i_n] += dequantized_tile_block

    # Dequantize matmul results using rhs scale and copying results from SBUF to HBM.
    rhs_scale_sbuf = nl.ndarray(rhs_scale.shape, buffer=nl.sbuf, dtype=rhs_scale.dtype)
    rhs_scale_sbuf = nl.load(rhs_scale)

    i_result = nl.mgrid[0:TILE_M, 0:N]
    for bm in nl.affine_range(TILES_IN_BLOCK_M):
      result_tiles_i_k = i_result.p
      result_tiles_i_n = bm * (NUM_BLOCK_N * BLOCK_N) + i_result.x

      result_i_m = m * BLOCK_M + bm * TILE_M + i_result.p
      result_i_n = i_result.x

      # FIXME: remove after dequantizing using nisa.scalar_tensor_tensor for dequantization.
      dequantized = nisa.tensor_tensor(
        result_tiles[result_tiles_i_k, result_tiles_i_n],
        rhs_scale_sbuf,
        nl.multiply
      )

      nl.store(result[result_i_m, result_i_n], value=dequantized)
    
  return result
