"""
Batch-Invariant MatMul Kernel

This kernel demonstrates batch invariance in matrix multiplication by controlling
the M-dimension tiling strategy.
"""

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

@nki.compiler.skip_middle_end_transformations
@nki.jit
def nki_matmul_kernel_isa(a, b, batch_invariant=True):
    """
    Matrix multiplication with batch invariance parameter
    
    batch_invariant=True:  Uses K_TILE=128 
    batch_invariant=False: Dynamic K_TILE size used
    
    This demonstrates how different K tiling affects numerical results.
    """
    K, M = a.shape
    N = b.shape[1]
    M_TILE = 128
    
    # ONLY DIFFERENCE: K_TILE strategy
    if batch_invariant:
        K_TILE = 128  # Always hardcoded
    else:
        K_TILE = 64 if K <= 512 else 128  # Adaptive

    result = nl.ndarray((M, N), dtype=a.dtype, buffer=nl.shared_hbm)
    
    for m in nl.affine_range(M // M_TILE):
        # Accumulator for this M chunk
        c_psum = nl.zeros((M_TILE, N), dtype=nl.float32, buffer=nl.psum)
        
        # Reduction over K
        for k in nl.affine_range(K // K_TILE):
            # Load a: [K_TILE, M_TILE]
            i_a_p, i_a_f = nl.mgrid[0:K_TILE, 0:M_TILE]
            a_tile = nl.load(a[k*K_TILE + i_a_p, m*M_TILE + i_a_f])
            
            # Load b: [K_TILE, N]
            i_b_p, i_b_f = nl.mgrid[0:K_TILE, 0:N]
            b_tile = nl.load(b[k*K_TILE + i_b_p, i_b_f])
            
            # Matmul

            print(a_tile.shape, b_tile.shape)
            c_psum += nisa.nc_matmul(a_tile, b_tile)
        # Store this M chunk
        i_out_p, i_out_f = nl.mgrid[0:M_TILE, 0:N]
        c_sbuf = nl.copy(c_psum, dtype=result.dtype)
        nl.store(result[m*M_TILE + i_out_p, i_out_f], value=c_sbuf)
    
    return result

@nki.jit
def nki_matmul_kernel_lang(a, b, batch_invariant=True):
    """
    Matrix multiplication with batch invariance parameter
    
    batch_invariant=True:  Uses K_TILE=128 
    batch_invariant=False: Uses K_TILE=64  
    
    This demonstrates how different K tiling affects numerical results.
    """
    M, K = a.shape
    N = b.shape[1]
    M_TILE = 128
    
    # ONLY DIFFERENCE: K_TILE strategy
    if batch_invariant:
        K_TILE = 128  # Always hardcoded
    else:
        K_TILE = 64 if K <= 512 else 128  # Adaptive
    
    result = nl.ndarray((M, N), dtype=a.dtype, buffer=nl.shared_hbm)
    
    for m in nl.affine_range(M // M_TILE):
        # Accumulator for this M chunk
        c_psum = nl.zeros((M_TILE, N), dtype=nl.float32, buffer=nl.psum)
        
        # Reduction over K
        for k in nl.affine_range(K // K_TILE):
            # Load a: [M_TILE, K_TILE]
            i_a_p, i_a_f = nl.mgrid[0:M_TILE, 0:K_TILE]
            a_tile = nl.load(a[m*M_TILE + i_a_p, k*K_TILE + i_a_f])
            
            # Load b: [K_TILE, N]
            i_b_p, i_b_f = nl.mgrid[0:K_TILE, 0:N]
            b_tile = nl.load(b[k*K_TILE + i_b_p, i_b_f])
            
            # Matmul
            c_psum += nl.matmul(a_tile, b_tile, transpose_x=False)
        
        # Store this M chunk
        i_out_p, i_out_f = nl.mgrid[0:M_TILE, 0:N]
        c_sbuf = nl.copy(c_psum, dtype=result.dtype)
        nl.store(result[m*M_TILE + i_out_p, i_out_f], value=c_sbuf)
    
    return result
