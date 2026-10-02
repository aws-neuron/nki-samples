"""
Copyright (C) 2026, Amazon.com. All Rights Reserved

NKI implementation for tensor indirection NKI tutorial.

"""
# NKI_EXAMPLE_42_BEGIN
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def tensor_indirection_kernel(index_tensor, data_tensor):
  """
  NKI implementation for tensor indirection over the data tensor's free dim.

  This kernel gathers columns from each row of the data tensor using a list of
  free-dim index offsets. It uses ``data_tile.indirect(idx_tile)`` (TI) to do the
  gather in a single vector engine instruction. The index row is broadcast to
  every data row, so all partitions share the same gather pattern, and the
  kernel computes ``output = data_tensor[:, index_tensor]``.

  TI reads the idx tile column by column within each 16-partition group. In a
  group starting at row ``start``, output column ``i`` uses the offset stored at
  ``idx_tile[start + (i % 16), i // 16]``. The index_f offsets therefore occupy
  only ``(16, index_f // 16)`` slots per group.

  This pattern comes up in MoE expert routing. For each expert we gather the
  slice of the activation tensor that has been routed to it, and the same
  free-dim selection applies to every token row.

  Shape convention:
    index_tensor: (1, index_f)        uint16 free-dim offsets
    data_tensor:  (data_p, data_free) source to gather from
    output:       (data_p, index_f)   gathered values

  Build the idx tile fed to TI in three steps:
    1. Reshape (1, index_f) into (index_f // 16, 16) via per-row DMAs,
       making the 16 axis contiguous on the free dim.
    2. nc_transpose to (16, index_f // 16) so the 16 axis lives on
       partitions — the layout TI expects within each 16-partition group.
    3. Replicate that 16-partition tile across all data partitions, since
       every row gathers from the same set of free-dim offsets.

  Both index_f and data_p must be a multiple of 16; index_tensor's
  partition dim is assumed to be 1 for this demonstration.
  """
  index_p, index_f = index_tensor.shape
  assert index_p == 1, (
    f"Expect index_tensor partition dim to be 1, got {index_p}"
  )
  assert index_f % 16 == 0, (
    f"Expect index_tensor free dim to be a multiple of 16, got {index_f}"
  )

  data_p, _ = data_tensor.shape
  assert data_p % 16 == 0, (
    f"Expect data_tensor partition dim to be a multiple of 16, got {data_p}"
  )

  out_tensor = nl.ndarray((data_p, index_f),
                          dtype=data_tensor.dtype, buffer=nl.shared_hbm)

  # Load both the data and index tensors into SBUF.
  data_tile = nl.load(data_tensor)
  index_tile = nl.load(index_tensor)

  # Reshape [1, index_f] into [index_f // 16, 16] with one DMA per row.
  reshaped_tile = nl.ndarray((index_f // 16, 16), dtype=nl.uint16, buffer=nl.sbuf)
  for k in nl.static_range(index_f // 16):
    nisa.dma_copy(
      dst=reshaped_tile[k:k + 1, :],
      src=index_tile[0:1, k * 16:(k + 1) * 16],
    )

  # Transpose [index_f // 16, 16] to [16, index_f // 16] to move 16 onto partitions.
  transposed_tile = nl.ndarray((16, index_f // 16), dtype=nl.uint16, buffer=nl.sbuf)
  nisa.nc_transpose(dst=transposed_tile,
                    data=reshaped_tile, engine=nisa.engine.vector)

  # Replicate the 16-partition tile to fill all data_p partitions.
  idx_tile = nl.ndarray((data_p, index_f // 16), dtype=nl.uint16, buffer=nl.sbuf)
  for g in nl.static_range(data_p // 16):
    nisa.dma_copy(dst=idx_tile[16 * g:16 * (g + 1), :], src=transposed_tile)

  # Gather index_f free-dim columns per row via tensor indirection.
  gathered_tile = data_tile.indirect(idx_tile, num_elem=index_f)
  result_tile = nl.ndarray((data_p, index_f),
                           dtype=data_tensor.dtype, buffer=nl.sbuf)
  nisa.tensor_copy(result_tile, gathered_tile, engine=nisa.engine.vector)

  nl.store(out_tensor, value=result_tile)

  return out_tensor
  # NKI_EXAMPLE_42_END
