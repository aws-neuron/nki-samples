"""
Copyright (C) 2026, Amazon.com. All Rights Reserved

NKI implementation for average pool 2D NKI tutorial.

"""
import numpy as np
# NKI_EXAMPLE_37_BEGIN
import nki
import nki.isa as nisa
import nki.language as nl
from nki.typing import tensor

@nki.jit
def tensor_avgpool_kernel(in_tensor, pool_size):
  """NKI kernel to compute a 2D avg-pool operation

  Args:
      in_tensor: an input tensor, of shape C x H x W
      pool_size: an integer representing a (square) pool-window size

  Return:
      out_tensor: the resulting output tensor, of shape C x (H/pool_size) x (W/pool_size)
  """
  ssert pool_size >= 1, "pool_size must be >= 1"

  # Get input/output dimensions
  sz_cin, sz_hin, sz_win = in_tensor.shape
  sz_hout = sz_hin // pool_size
  sz_wout = sz_win // pool_size
  # Create output tensor shared between all SPMD instances as result tensor
  out_tensor = nl.ndarray((sz_cin, sz_hout, sz_wout), dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)

  # Set relevant sizes
  sz_p = sz_cin
  sz_pool = pool_size

  # Use an access pattern to create a 5D view of the input:
  # [sz_p, sz_hout, sz_wout, sz_pool, sz_pool]
  # The pool dimensions are placed last so we can reduce over them.

  # Load input data from external memory to on-chip memory
  in_tile = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype, buffer=nl.sbuf)
  nisa.dma_copy(dst=in_tile, src=in_tensor)

  # Perform the pooling operation using an access pattern view:
  # The .ap() creates a strided 5D view of the 3D input tile,
  # grouping elements into pool windows for reduction.
  pool_view = in_tile.ap([
    [sz_hin * sz_win, sz_p],      # partition stride
    [sz_pool * sz_win, sz_hin // sz_pool],  # outer row stride
    [sz_pool, sz_win // sz_pool],            # outer col stride
    [sz_win, sz_pool],             # inner row stride (within pool window)
    [1, sz_pool],                  # inner col stride (within pool window)
  ])
  sum_tile = nl.sum(pool_view, axis=[3, 4])
  out_tile = nl.ndarray(sum_tile.shape, dtype=sum_tile.dtype, buffer=nl.sbuf)
  nisa.tensor_scalar(dst=out_tile, data=sum_tile, op0=nl.multiply,
                     operand0=1.0 / (pool_size * pool_size))

  # Store the results back to hbm
  nisa.dma_copy(dst=out_tensor, src=out_tile)

  # Transfer the ownership of `out_tensor` to the caller
  return out_tensor
  # NKI_EXAMPLE_37_END


# Reference NumPy implementation
def np_average_pool_2D(in_tensor, pool_size):
  c, h_in, w_in = in_tensor.shape
  reshaped = in_tensor.reshape(c, h_in // pool_size, pool_size, w_in // pool_size, pool_size)
  return np.nanmean(reshaped, axis=(2, 4))


if __name__ == "__main__":
  # Now let's run the kernel
  POOL_SIZE = 2
  C, HIN, WIN = 2, 6, 6
  HOUT, WOUT = HIN//POOL_SIZE, WIN//POOL_SIZE

  in_tensor = np.arange(C * HIN * WIN, dtype=np.float16).reshape(C, HIN, WIN)

  out_nki = tensor_avgpool_kernel(in_tensor, POOL_SIZE)

  out_np = np_average_pool_2D(in_tensor, POOL_SIZE)

  print(in_tensor, out_nki, out_np)

  match = (out_nki == out_np).all()

  if match:
    print("NKI and NumPy match")
  else:
    print("NKI and NumPy differ")

  assert match
