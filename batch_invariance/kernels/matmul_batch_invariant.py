"""
Batch-Invariant MatMul Kernel

This kernel demonstrates batch invariance in matrix multiplication by controlling
the K-dimension tiling strategy.

NKI version: 0.3.0 (Beta 3)
"""

import nki
import nki.isa as nisa
import nki.language as nl
import nki.typing as nt


@nki.jit
def nki_matmul_kernel_isa(a, b, deterministic=True):
    """
    Matrix multiplication with batch invariance parameter.

    Args:
        a: Input matrix of shape [K, M]
        b: Input matrix of shape [K, N]
        deterministic: If True, uses fixed K_TILE=128 regardless of K size,
                       producing identical results across different batch sizes.
                       If False, uses K_TILE=64 (more accumulations, different rounding).

    Returns:
        result: Output matrix of shape [M, N], same dtype as inputs

    Notes:
        PSUM always accumulates in float32 regardless of input dtype.
        The ONLY difference between modes is K_TILE size. Different K_TILE sizes
        change the number and order of float32 accumulations in PSUM, which can
        produce slightly different results due to non-associativity of FP arithmetic.
        With bfloat16 inputs this difference vanishes (invariant); with float32 it does not.
    """
    K, M = a.shape
    N = b.shape[1]
    M_TILE = 128

    # ONLY DIFFERENCE: K_TILE strategy (must be ≤128: partition dim constraint on stationary/moving)
    if deterministic:
        K_TILE = min(128, K)  # Always hardcoded — same accumulation count regardless of K
    else:
        K_TILE = min(64, K)   # Smaller tiles → more accumulations → different rounding

    assert K % K_TILE == 0, f"K={K} must be divisible by K_TILE={K_TILE}"

    result = nl.ndarray((M, N), dtype=a.dtype, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // M_TILE):
        # PSUM always accumulates in float32 regardless of input dtype
        c_psum = nl.zeros((M_TILE, N), dtype=nl.float32, buffer=nl.psum)

        for k in nl.affine_range(K // K_TILE):
            a_start = k * K_TILE
            a_end = min(K, a_start + K_TILE)
            m_start = m * M_TILE
            m_end = min(M, m_start + M_TILE)

            a_tile = nl.ndarray((K_TILE, M_TILE), dtype=a.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=a_tile, src=a[a_start:a_end, m_start:m_end])

            b_start = k * K_TILE
            b_end = min(K, b_start + K_TILE)
            b_tile = nl.ndarray((K_TILE, N), dtype=b.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=b_tile, src=b[b_start:b_end, 0:N])

            # Matmul — multiple writes to same c_psum trigger hardware accumulation
            nisa.nc_matmul(dst=c_psum, stationary=a_tile, moving=b_tile)

        # Copy PSUM (float32) -> SBUF (input dtype), then DMA to HBM
        c_sbuf = nl.ndarray((M_TILE, N), dtype=a.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=c_sbuf, src=c_psum)

        c_start = m * M_TILE
        c_end = min(M, c_start + M_TILE)
        nisa.dma_copy(dst=result[c_start:c_end, 0:N], src=c_sbuf)

    return result
