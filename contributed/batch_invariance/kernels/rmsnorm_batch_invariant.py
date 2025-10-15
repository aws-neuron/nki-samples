"""
Batch-Invariant RMSNorm Kernel
"""

import math
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl


@nki.jit
def nki_rmsnorm_kernel(a_tensor, g_tensor, batch_invariant=True):
  """
  RMSNorm with batch invariance parameter
  
  This demonstrates TRUE batch invariance testing:
  - batch_invariant=True: Always uses tile_size=128 (same strategy regardless of batch)
  - batch_invariant=False: Adapts tile_size based on batch size (different strategies)
  """
  out_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype,
                          buffer=nl.shared_hbm)

  # Make sure shapes match
  assert a_tensor.shape[1] == g_tensor.shape[0]

  num_rows = a_tensor.shape[0]
  hidden_dim = a_tensor.shape[1]
  
  # CRITICAL: Tile size based on BATCH SIZE (not hidden_dim)
  # This is what creates batch variance!
  if batch_invariant:
    # INVARIANT: Fixed strategy regardless of batch size
    tile_size = 128
  else:
    # VARIANT: Strategy changes based on batch size
    # Small batches get smaller tiles -> different processing pattern
    if num_rows <= 64:
      tile_size = 32  # Small batch: smaller tiles
    else:
      tile_size = 128  # Large batch: larger tiles
  
  # Generate tensor indices based on tile_size
  ix = nl.arange(tile_size)[:, None]
  iw = nl.arange(1)[:, None]
  iy = nl.arange(hidden_dim)[None, :]

  # Load RMSNorm weight once
  g_tile = nl.load(g_tensor.reshape((1, hidden_dim))[iw, iy])

  # Process tile_size rows at a time
  for i in nl.affine_range(math.ceil(num_rows / tile_size)):

    # Load input data from external memory to on-chip memory
    a_tile = nl.load(a_tensor[i * tile_size + ix, iy],
                    mask=(i * tile_size + ix < num_rows))

    # Compute element-wise square of a_tensor
    in_square = nl.square(a_tile)

    # Calculate sum of squared elements, along last dimension
    square_sum = nl.sum(in_square, axis=[1])

    # Scale and get a reciprocal
    mean = square_sum / hidden_dim

    # Take square root of mean and then reciprocal with rsqrt API
    rms_reciprocal = nl.rsqrt(mean)

    # Scale the input tensor
    out_tile = nl.multiply(a_tile, rms_reciprocal)

    # Broadcast weight along first axis to match tensor shape
    g_bcast = g_tile.broadcast_to((tile_size, hidden_dim))

    # Multiply with the RMSNorm weight
    out_tile[...] = nl.multiply(out_tile, g_bcast,
                           mask=(i * tile_size + ix < num_rows))

    # store the results back to external memory
    nl.store(out_tensor[i * tile_size + ix, iy], value=out_tile,
            mask=(i * tile_size + ix < num_rows))

  return out_tensor