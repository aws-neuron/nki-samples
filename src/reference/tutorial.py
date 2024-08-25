"""
Copyright (c) 2023, Amazon.com. All Rights Reserved

kernels - Builtin high performance NKI kernels used in tutorial

"""

import neuronxcc.nki.language as nl

def add_kernel_nx8x128x512(a_ptr, b_ptr, c_ptr, n_elements):
  ix = nl.arange(128)[:, None]
  iy = nl.arange(512)[None, :]

  tile_size = 128 * 512
  block_size = 8 * tile_size

  j = nl.program_id(axis=0)

  for i in nl.affine_range(8):
    offset = j * block_size + i * tile_size + 512 * ix + iy
    mask = offset < n_elements
    a_ptr = a_ptr.ptr + offset
    b_ptr = b_ptr.ptr + offset
    c_ptr = c_ptr.ptr + offset

    a = nl.load(a_ptr, mask=mask)
    b = nl.load(b_ptr, mask=mask)
    c = a + b
    nl.store(c_ptr, value=c, mask=mask)