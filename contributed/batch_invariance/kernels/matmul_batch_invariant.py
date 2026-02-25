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
            nisa.dma_copy(
                dst=a_tile,
                src=a[k*K_TILE : (k+1)*K_TILE, m*M_TILE : (m+1)*M_TILE]
            )
            
            # Allocate and load b: [K_TILE, N]
            b_tile = nl.ndarray((K_TILE, N), dtype=b.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=b_tile,
                src=b[k*K_TILE : (k+1)*K_TILE, 0:N]
            )
            # Matmul
            c_psum += nisa.nc_matmul(a_tile, b_tile)
        
        # Store this M chunk
        c_sbuf = nl.ndarray((M_TILE, N), dtype=result.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=c_sbuf, src=c_psum)
        nisa.dma_copy(
            dst=result[m*M_TILE : (m+1)*M_TILE, 0:N],
            src=c_sbuf
        )
    
    return result
