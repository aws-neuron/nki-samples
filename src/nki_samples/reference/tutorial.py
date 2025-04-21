"""
Copyright (c) 2023, Amazon.com. All Rights Reserved

kernels - Builtin high performance NKI kernels used in tutorial

"""

from neuronxcc import nki
import neuronxcc.nki.language as nl


@nki.jit
def add_kernel_nx8x128x512(a_ptr, b_ptr, n_elements):
  c_ptr = nl.ndarray(a_ptr.shape, dtype=a_ptr.dtype, buffer=nl.shared_hbm)

  ix, iy = nl.mgrid[0:128, 0:512]

  tile_size = 128 * 512
  block_size = 8 * tile_size

  j = nl.program_id(axis=0)

  for i in nl.affine_range(8):
    offset = j * block_size + i * tile_size + 512 * ix + iy
    a = nl.load(a_ptr[j, i, ix, iy], mask=offset < n_elements)
    b = nl.load(b_ptr[j, i, ix, iy], mask=offset < n_elements)
    c = nl.add(a, b, mask=offset < n_elements)
    nl.store(c_ptr[j, i, ix, iy], value=c, mask=offset < n_elements)

  return c_ptr
