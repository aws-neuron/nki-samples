"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

Kernels written by the AWS Neuron.

WARNING: These kernels:
   - Are tested only against internal nightly builds
   - May not be compatible with public NeuronSDK releases
   - Have not been extensively tested across all input configurations
   - Carry no compatibility guarantees
   - The behavior of these kernels may be modified without prior notice

kernels - hardware microbenchmarks

"""
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from neuronxcc import nki
import neuronxcc.nki as nki
from neuronxcc.nki._private.private_api import row_tiled_mm_kernel, column_tiled_mm_kernel

@nki.jit
def row_tiled_matmul_isa_kernel(A, B, n_tile, tiling_factor, use_full_pe):
  _, k, m = A.shape
  _, _, n = B.shape
  out = nl.ndarray((tiling_factor, m, n), dtype=A.dtype, buffer=nl.shared_hbm)

  row_tiled_mm_kernel("RowTiledMM", A, B, out, n_tile=n_tile, tiling_factor=tiling_factor, use_full_pe=use_full_pe)
  return out


@nki.jit
def column_tiled_matmul_isa_kernel(A, B, n_tile, use_full_pe):
  k, m = A.shape
  _, n = B.shape
  out = nl.ndarray((m, n), dtype=A.dtype, buffer=nl.shared_hbm)

  column_tiled_mm_kernel("ColumnTiledMM", A, B, out, n_tile=n_tile, use_full_pe=use_full_pe)
  return out
