"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

Kernels written by the AWS Neuron Team.

WARNING: These kernels:
   - Are tested only against internal nightly compiler builds
   - May rely on internal compiler feature/flags and not be compatible with public NeuronSDK
   - Have not been extensively tested across all input configurations
   - Carry no compatibility guarantees
   - The behavior of these kernels may be modified without prior notice

Flash Paged Attention kernels with variable-length sequence inputs.

"""

import numpy as np
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc.nki.language import par_dim
from neuronxcc.nki.isa.constants import oob_mode

from utils import (
    B_P_SIZE,
    ceil_div,
    is_power_of_2,
    transform_to_vector_dge_layout,
    broadcast_partition_with_PE,
    PF_transpose_with_PE,
)


def prepare_kv_block_dim_tiling(key_cache, value_cache, LARGE_KV_TILE_SIZE):
    """
    If number of blocks to load for a tile (i.e. LARGE_KV_TILE_SIZE // block_size) is smaller than
    B_P_SIZE(128), tiling on block_size dimension is applied so that there are 128 loads to fully
    utilize Vector DGE.

    This function decides the new tiled_block_size. It also reshapes KV cache to a 2-D layout to
    load KV data in block granularity
    """
    num_blocks, k_h, block_size, d = key_cache.shape
    num_blocks_per_large_tile = LARGE_KV_TILE_SIZE // block_size
    assert is_power_of_2(
        num_blocks_per_large_tile
    ), f"{num_blocks_per_large_tile=} is expected of be power of 2"

    # load block tables
    if num_blocks_per_large_tile < B_P_SIZE:
        # we checked num_blocks_per_tile is a power of 2
        assert B_P_SIZE % num_blocks_per_large_tile == 0
        block_size_tiling_factor = B_P_SIZE // num_blocks_per_large_tile
        assert block_size % block_size_tiling_factor == 0
        num_blocks_per_large_tile *= block_size_tiling_factor  # i.e. = B_P_SIZE
    else:
        block_size_tiling_factor = 1
    tiled_block_size = block_size // block_size_tiling_factor

    # flatten KV cache to be 2D for loading into SBUF
    new_cache_shape = (
        num_blocks * k_h * block_size_tiling_factor,
        tiled_block_size * d,
    )
    key_cache = key_cache.reshape(new_cache_shape)
    value_cache = value_cache.reshape(new_cache_shape)
    return key_cache, value_cache, block_size_tiling_factor, tiled_block_size


def transform_block_tables_for_indirect_load(
    block_tables,
    block_size_tiling_factor,
    num_head,
    head_id,
):
    """
    This function calculates the new block ids after reshaping KV cache layout from [num_blocks,
    k_h, block_size, d] to [num_blocks * k_h, bloock_size * d].

    And then block_tables is transposed (from [num_tiles, num_blocks_per_tile] to
    [num_blocks_per_tile, num_tiles]) to map block_ids per tile to SBUF Partition Dimension for
    vector DGE.
    """
    # block_tables on HBM has layout [num_tiles, num_blocks_per_tile]. And after loaded to SBUF,
    # the layout becomes [par_dim(128), ceil_div(num_tiles, 128), num_blocks_per_tile]
    num_tiles_per_partition, num_partitions, num_blocks_per_tile = block_tables.shape

    num_loads = ceil_div(num_blocks_per_tile, B_P_SIZE)
    block_tables_transposed = nl.ndarray(
        (par_dim(B_P_SIZE), num_loads, num_partitions * num_tiles_per_partition),
        dtype=nl.int32,
    )

    # prepare iota ahead of time to avoid repeatedly using Gpsimd
    if num_head > 1:
        # helper func may not properly broadcast int32, need testing
        head_id_0 = nisa.iota(head_id, dtype=nl.uint32).reshape((1, 1))
        if num_tiles_per_partition > 1:
            head_id = nl.ndarray(
                (par_dim(num_tiles_per_partition), 1),
                dtype=nl.int32,
            )
            broadcast_partition_with_PE(head_id_0, head_id, out_in_psum=False)
        else:
            head_id = head_id_0
        if num_blocks_per_tile > 1:
            head_id = head_id.broadcast_to(
                (num_tiles_per_partition, num_blocks_per_tile)
            )

    if block_size_tiling_factor > 1:
        broadcast_shape = (
            num_tiles_per_partition,
            num_blocks_per_tile,
            block_size_tiling_factor,
        )
        offset = nisa.iota(
            nl.arange(block_size_tiling_factor)[None, None, :], dtype=nl.int32
        )
        if num_tiles_per_partition > 1 or num_blocks_per_tile > 1:
            offset = offset.broadcast_to(broadcast_shape)

    block_tables_transposed_reshaped = block_tables_transposed.reshape(
        (par_dim(B_P_SIZE), num_loads, num_partitions, num_tiles_per_partition)
    )
    identity_for_transpose_hbm = nl.shared_constant(
        np.identity(n=num_tiles_per_partition, dtype=np.uint8),
        dtype=nl.uint8,
    )
    identity_for_transpose = nl.load(identity_for_transpose_hbm)
    for partition_id in nl.affine_range(num_partitions):
        block_tables_partition = block_tables[:, partition_id]
        if num_head > 1:
            # fuse num_block and num_head dimension
            block_tables_partition = block_tables_partition * num_head
            block_tables_partition = nisa.tensor_tensor(
                block_tables_partition,
                head_id,
                nl.add,
                dtype=nl.int32,
            )

        # tile block size dimension
        if block_size_tiling_factor > 1:
            assert num_blocks_per_tile * block_size_tiling_factor == B_P_SIZE
            block_tables_partition = (
                (block_tables_partition * block_size_tiling_factor)
                .reshape((num_tiles_per_partition, num_blocks_per_tile, 1))
                .broadcast_to(broadcast_shape)
            )
            new_block_tables = block_tables_partition + offset
            new_block_tables = new_block_tables.reshape(
                (num_tiles_per_partition, B_P_SIZE)
            )
        else:
            new_block_tables = block_tables_partition

        # transpose the block table so that it can be used by vector DGE
        transform_to_vector_dge_layout(
            indices_in=new_block_tables,
            indices_out=block_tables_transposed_reshaped[:, :, partition_id, :],
            identity_for_transpose=identity_for_transpose,
        )
    return block_tables_transposed


def load_kv_tile_from_cache(
    key_cache,
    value_cache,
    block_tables,
    large_k_tile_idx,
    num_blocks_per_large_tile,
    block_size,
    B_D_SIZE,
    kernel_dtype,
    k_load_buffer,
    v_load_buffer,
    identity_for_transpose,
):
    # load key cache
    assert num_blocks_per_large_tile % B_P_SIZE == 0
    num_loads = num_blocks_per_large_tile // B_P_SIZE
    LARGE_KV_TILE_SIZE = num_blocks_per_large_tile * block_size
    transposed_k_tile = nl.ndarray(
        (par_dim(B_D_SIZE), LARGE_KV_TILE_SIZE),
        dtype=kernel_dtype,
    )
    transposed_k_tile_reshape = transposed_k_tile.reshape(
        (par_dim(B_D_SIZE), num_loads, block_size, B_P_SIZE)
    )
    for load_idx in nl.affine_range(num_loads):
        i_p = nl.arange(B_P_SIZE)[:, None]
        i_f = nl.arange(block_size * B_D_SIZE)[None, :]
        k_load_buffer[i_p, load_idx, i_f] = nl.load(
            key_cache[block_tables[i_p, load_idx, large_k_tile_idx], i_f],
            mode=oob_mode.skip,
        )

        # Transpose SBUF tensor using PE
        for tb_i in nl.affine_range(block_size):
            if key_cache.dtype != kernel_dtype:
                k_src = nl.copy(
                    k_load_buffer[:, load_idx, nl.ds(tb_i * B_D_SIZE, B_D_SIZE)],
                    dtype=kernel_dtype,
                )
            else:
                k_src = k_load_buffer[:, load_idx, nl.ds(tb_i * B_D_SIZE, B_D_SIZE)]
            PF_transpose_with_PE(
                src=k_src,
                out=transposed_k_tile_reshape[
                    :,
                    load_idx,
                    tb_i,
                ],
                identity_for_transpose=identity_for_transpose,
            )
    # load value cache
    for load_idx in nl.affine_range(num_loads):
        i_p = nl.arange(B_P_SIZE)[:, None]
        i_f = nl.arange(block_size * B_D_SIZE)[None, :]
        v_load_buffer[i_p, i_f + load_idx * block_size * B_D_SIZE] = nl.load(
            value_cache[block_tables[i_p, load_idx, large_k_tile_idx], i_f],
            mode=oob_mode.skip,
        )
    if kernel_dtype != v_load_buffer.dtype:
        v_tile = nl.ndarray(
            v_load_buffer.shape,
            dtype=kernel_dtype,
        )
        v_tile[...] = nl.copy(v_load_buffer, dtype=kernel_dtype)
    else:
        v_tile = v_load_buffer
    return transposed_k_tile, v_tile
