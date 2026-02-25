"""
RMSNorm to demonstrate Batch Variance

This kernel tiles the HIDDEN DIMENSION (reduction axis) instead of just the batch dimension.
This creates different accumulation orders and breaks batch-invariance!
"""

import math
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_rmsnorm_kernel_isa(a, g, deterministic=True):
    """
    RMSNorm with split reduction along hidden dimension
    
    deterministic=True:  HIDDEN_TILE=256 (fewer chunks, fewer accumulations)
    deterministic=False: HIDDEN_TILE=128 (more chunks, more accumulations)
    
    This demonstrates REAL batch variance because different tile sizes
    change the order of floating-point additions during reduction.
    """
    out_tensor = nl.ndarray(a.shape, dtype=a.dtype,
                            buffer=nl.shared_hbm)
    
    assert a.shape[1] == g.shape[0]
    
    num_rows = a.shape[0]
    hidden_dim = a.shape[1]
    BATCH_TILE = 128
    
    # CRITICAL: Tile size for REDUCTION dimension (hidden_dim)
    # Different sizes = different number of accumulations = variance!
    if deterministic:
        HIDDEN_TILE = 128  # Fixed - same accumulation order always
    else:
        HIDDEN_TILE = min(64, hidden_dim) if hidden_dim <= 256 else (128 if hidden_dim <= 512 else 256)  # Adaptive
    
    # Load weight once using nisa.dma_copy
    g_tile = nl.ndarray((1, hidden_dim), dtype=g.dtype, buffer=nl.sbuf)
    g = g.reshape((1, hidden_dim))
    nisa.dma_copy(
        src=g[0:1, 0:hidden_dim],
        dst=g_tile[0:1, 0:hidden_dim]
    )

    # Loop over batch dimension
    for i in nl.affine_range(math.ceil(num_rows / BATCH_TILE)):
        # SPLIT REDUCTION: Accumulate partial sums across hidden dimension chunks
        partial_square_sum = nl.ndarray((BATCH_TILE, 1), dtype=nl.float32, buffer=nl.psum)
        a_start = i * BATCH_TILE
        a_end = min(num_rows, a_start + BATCH_TILE)
        # Iterate over hidden dimension in chunks
        for h in nl.affine_range(math.ceil(hidden_dim / HIDDEN_TILE)):
            # Allocate buffer for chunk
            a_tile = nl.ndarray((BATCH_TILE, HIDDEN_TILE), dtype=a.dtype, buffer=nl.sbuf)
            
            # Load chunk with mask using nisa.dma_copy
            
            h_start = h * HIDDEN_TILE
            h_end = min(hidden_dim, h_start + HIDDEN_TILE)
            nisa.dma_copy(
                src=a[a_start:a_end, h_start:h_end],
                dst=a_tile,
            )
            
            # Square this chunk
            chunk_square = nl.square(a_tile)
            
            # Reduce this chunk (sum along hidden dimension) using nisa.tensor_reduce
            chunk_sum = nisa.tensor_reduce(
                nl.add, 
                chunk_square, 
                axis=[1], 
                keepdims=True,
                dtype=nl.float32
            )
            
            # ACCUMULATE: This is where variance enters!
            # Different HIDDEN_TILE sizes mean different number of additions
            partial_square_sum += chunk_sum
        
        # Compute mean and RMS
        mean = partial_square_sum * (1.0 / hidden_dim)
        rms_reciprocal = nl.rsqrt(mean)
        
        # Allocate buffer for full tile
        a_tile = nl.ndarray((BATCH_TILE, hidden_dim), dtype=a.dtype, buffer=nl.sbuf)
        
        # Load full row for normalization using nisa.dma_copy
        nisa.dma_copy(
            src=a[a_start:a_end, :],
            dst=a_tile,
        )
        
        # Normalize by RMS
        out_tile = nl.multiply(a_tile, rms_reciprocal)
        
        # Apply weight
        g_bcast = g_tile.broadcast_to((BATCH_TILE, hidden_dim))
        out_tile = nl.multiply(out_tile, g_bcast)
        
        # Store result using nisa.dma_copy
        nisa.dma_copy(
            src=out_tile,
            dst=out_tensor[a_start:a_end, :],
        )

    return out_tensor
