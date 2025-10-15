"""
RMSNorm with Split Reduction - Demonstrates TRUE Batch Variance

This kernel tiles the HIDDEN DIMENSION (reduction axis) instead of just the batch dimension.
This creates different accumulation orders and breaks batch-invariance!
"""

import math
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl


@nki.jit
def nki_rmsnorm_split_reduction(a_tensor, g_tensor, batch_invariant=True):
    """
    RMSNorm with split reduction along hidden dimension
    
    batch_invariant=True:  HIDDEN_TILE=256 (fewer chunks, fewer accumulations)
    batch_invariant=False: HIDDEN_TILE=128 (more chunks, more accumulations)
    
    This demonstrates REAL batch variance because different tile sizes
    change the order of floating-point additions during reduction.
    """
    out_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype,
                            buffer=nl.shared_hbm)
    
    assert a_tensor.shape[1] == g_tensor.shape[0]
    
    num_rows = a_tensor.shape[0]
    hidden_dim = a_tensor.shape[1]
    BATCH_TILE = 128
    
    # CRITICAL: Tile size for REDUCTION dimension (hidden_dim)
    # Different sizes = different number of accumulations = variance!
    if batch_invariant:
        HIDDEN_TILE = 256  # Fewer chunks (e.g., 2 for hidden_dim=512)
    else:
        HIDDEN_TILE = 128  # More chunks (e.g., 4 for hidden_dim=512)
    
    ix = nl.arange(BATCH_TILE)[:, None]
    iw = nl.arange(1)[:, None]
    
    # Process batch in tiles
    for i in nl.affine_range(math.ceil(num_rows / BATCH_TILE)):
        
        # SPLIT REDUCTION: Accumulate partial sums across hidden dimension chunks
        # Use PSUM for accumulation (always float32 internally)
        partial_square_sum = nl.zeros((BATCH_TILE, 1), dtype=nl.float32, buffer=nl.psum)
        
        # Iterate over hidden dimension in chunks
        num_hidden_tiles = math.ceil(hidden_dim / HIDDEN_TILE)
        for h in nl.affine_range(num_hidden_tiles):
            h_start = h * HIDDEN_TILE
            
            # Create indices for this hidden chunk (always use full HIDDEN_TILE, mask later)
            iy = nl.arange(HIDDEN_TILE)[None, :]
            
            # Create mask for valid hidden indices
            valid_mask = ((i * BATCH_TILE + ix < num_rows) & 
                         (h * HIDDEN_TILE + iy < hidden_dim))
            
            # Load a CHUNK of the hidden dimension with proper indexing
            a_chunk = nl.load(
                a_tensor[i * BATCH_TILE + ix, h * HIDDEN_TILE + iy],
                mask=valid_mask
            )
            
            # Square this chunk
            in_square_chunk = nl.square(a_chunk)
            
            # Reduce this chunk (sum along hidden dimension)
            # Mask ensures we only sum valid elements
            chunk_sum = nl.sum(in_square_chunk, axis=[1], keepdims=True, 
                              mask=valid_mask)
            
            # ACCUMULATE: This is where variance enters!
            # Different HIDDEN_TILE sizes mean different number of additions
            partial_square_sum += chunk_sum
        
        # Compute mean and RMS
        mean = partial_square_sum / hidden_dim
        rms_reciprocal = nl.rsqrt(mean)
        
        # Now load full row for normalization
        iy_full = nl.arange(hidden_dim)[None, :]
        a_tile = nl.load(
            a_tensor[i * BATCH_TILE + ix, iy_full],
            mask=(i * BATCH_TILE + ix < num_rows)
        )
        
        # Normalize by RMS
        out_tile = nl.multiply(a_tile, rms_reciprocal)
        
        # Apply weight
        g_tile = nl.load(g_tensor.reshape((1, hidden_dim))[iw, iy_full])
        g_bcast = g_tile.broadcast_to((BATCH_TILE, hidden_dim))
        out_tile = nl.multiply(out_tile, g_bcast,
                              mask=(i * BATCH_TILE + ix < num_rows))
        
        # Store result
        nl.store(out_tensor[i * BATCH_TILE + ix, iy_full], value=out_tile,
                mask=(i * BATCH_TILE + ix < num_rows))
    
    return out_tensor
