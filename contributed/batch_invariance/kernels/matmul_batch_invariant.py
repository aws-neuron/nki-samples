"""
Batch-Invariant MatMul Kernel

This kernel demonstrates batch invariance in matrix multiplication by controlling
the M-dimension tiling strategy.
"""

import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def nki_matmul_kernel_isa(a, b, deterministic=True):
    """
    Matrix multiplication with batch invariance parameter
    
    deterministic=True:  Uses K_TILE=128 
    deterministic=False: Dynamic K_TILE size used
    
    This demonstrates how different K tiling affects numerical results.
    """
    K, M = a.shape
    N = b.shape[1]
    M_TILE = 128
    
    # ONLY DIFFERENCE: K_TILE strategy
    if deterministic:
        K_TILE = 128  # Always hardcoded
    else:
        K_TILE = 64 if K <= 512 else 512 # Adaptive

    result = nl.ndarray((M, N), dtype=a.dtype, buffer=nl.shared_hbm)
    
    for m in nl.affine_range(M // M_TILE):
        # Accumulator for this M chunk
        c_psum = nl.ndarray((M_TILE, N), dtype=nl.float32, buffer=nl.psum)
        # Reduction over K
        for k in nl.affine_range(K // K_TILE):
            # Allocate and load a: [K_TILE, M_TILE]
            a_tile = nl.ndarray((K_TILE, M_TILE), dtype=a.dtype, buffer=nl.sbuf)
            a_start = k*K_TILE
            a_end = min(K, a_start + K_TILE)

            m_start = m*M_TILE
            m_end = min(M, m_start + M_TILE)

            nisa.dma_copy(
                src=a[a_start:a_end, m_start:m_end],
                dst=a_tile,
            )
            
            # Allocate and load b: [K_TILE, N]
            b_start = k*K_TILE
            b_end = min(K, b_start + K_TILE)

            b_tile = nl.ndarray((K_TILE, N), dtype=b.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                src=b[b_start:b_end, 0:N],
                dst=b_tile,
            )
            # Matmul
            nisa.nc_matmul(dst=c_psum, stationary=a_tile, moving=b_tile)
            # c_psum += nisa.nc_matmul(a_tile, b_tile)
        
        # Store this M chunk
        c_sbuf = nl.ndarray((M_TILE, N), dtype=result.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=c_sbuf, src=c_psum)

        c_start = m*M_TILE
        c_end = min(M, c_start + M_TILE)
        nisa.dma_copy(
            src=c_sbuf,
            dst=result[c_start:c_end, 0:N]
        )
    
    return result
