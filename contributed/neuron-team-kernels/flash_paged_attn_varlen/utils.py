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
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki.language import par_dim

B_P_SIZE = nl.tile_size.pmax
B_FMAX_SIZE = nl.tile_size.gemm_moving_fmax


def ceil_div(a, b):
    return (a + b - 1) // b


def pad_to_multiple(a, b):
    return ceil_div(a, b) * b


def is_power_of_2(x):
    return x > 0 and (x & (x - 1)) == 0


def load_indices(indices_hbm, partition_size=None):
    """
    Load a 2D indices array of shape [num_tiles, num_indices] from HBM to SBUF

    To map num_tiles to SBUF partition dimension, this function automatically partitions num_tiles
    with partition_size set to min(num_tiles, B_P_SIZE=128)

    Output SBUF tensor shape:
      [par_dim(partition_size), ceil_div(num_tiles, partition_size), num_indices]
    """
    num_tiles, num_indices = indices_hbm.shape
    if partition_size is None:
        partition_size = min(B_P_SIZE, num_tiles)
    else:
        assert partition_size <= B_P_SIZE, f"Expect {partition_size=} <= {B_P_SIZE=}"
    num_partitions = ceil_div(num_tiles, partition_size)
    indices_sbuf = nl.zeros(
        (par_dim(partition_size), num_partitions, num_indices),
        dtype=indices_hbm.dtype,
    )
    for i in nl.affine_range(num_partitions):
        i_p = nl.arange(partition_size)[:, None]
        i_f = nl.arange(num_indices)[None, :]
        indices_sbuf[i_p, i, i_f] = nl.load(
            indices_hbm[i_p + i * partition_size, i_f],
            mask=(i_p + i * partition_size < num_tiles),
        )
    return indices_sbuf


def load_indices_for_loop_step(indices_hbm, loop_index, step_size, partition_size=None):
    """
    Load a 2D indices array with dim 0 range [loop_index * step_size, (loop_index + 1) * size) from
    HBM with start offset to SBUF

    To map num_tiles to SBUF partition dimension, this function automatically partitions num_tiles
    with partition_size set to min(step_size, B_P_SIZE=128)

    Output SBUF tensor shape: [par_dim(partition_size), ceil_div(size, partition_size), num_indices]
    """

    _, num_indices = indices_hbm.shape
    if partition_size is None:
        partition_size = min(B_P_SIZE, step_size)
    else:
        assert partition_size <= B_P_SIZE, f"Expect {partition_size=} <= {B_P_SIZE=}"
    assert (
        step_size % partition_size == 0
    ), f"Expect {step_size=} % {partition_size=} == 0"
    num_partitions = step_size // partition_size

    assert loop_index.shape == (1, 1)
    base_addr = nl.ndarray((par_dim(partition_size), 1), dtype=nl.int32)
    broadcast_partition_with_PE(
        src=nl.multiply(loop_index, step_size, dtype=nl.uint32),
        out=base_addr,
        out_in_psum=False,
    )
    indices_sbuf = nl.ndarray(
        (par_dim(partition_size), num_partitions, num_indices),
        dtype=indices_hbm.dtype,
    )
    for i in nl.affine_range(num_partitions):
        offset = nisa.iota(
            nl.arange(partition_size)[None, :] + i * partition_size, dtype=nl.int32
        )
        offset_transposed = nl.ndarray((partition_size, 1), dtype=nl.int32)
        PF_transpose_with_PE_int4byte(src=offset, out=offset_transposed)
        start_offsets = nisa.tensor_tensor(base_addr, offset_transposed, op=nl.add)
        i_p = nl.arange(partition_size)[:, None]
        i_f = nl.arange(num_indices)[None, :]
        indices_sbuf[i_p, i, i_f] = nl.load(
            indices_hbm[start_offsets[i_p, 0], i_f],
        )
    return indices_sbuf


def transform_to_vector_dge_layout(
    indices_in, indices_out, partition_size=None, identity_for_transpose=None
):
    """
    Transpose an tile of shape [tile_size, num_indices] so that num_indices is mapped to partition
    dimension and perform partition with partition_size=min(num_indices, B_P_SIZE=128)

    indices_in:
      [par_dim(tile_size), num_indices]
    indices_out:
      [par_dim(partition_size), ceil_div(num_indices, partition_size), tile_size]
    """
    tile_size, num_indices = indices_in.shape
    if partition_size is None:
        partition_size = min(num_indices, B_P_SIZE)
    else:
        assert partition_size <= B_P_SIZE, f"Expect {partition_size=} <= {B_P_SIZE=}"
    num_partitions = ceil_div(num_indices, partition_size)
    assert indices_out.shape == (
        partition_size,
        num_partitions,
        tile_size,
    )
    for i in nl.affine_range(num_partitions):
        PF_transpose_with_PE(
            indices_in[:, nl.ds(i * partition_size, partition_size)],
            indices_out[:, i, :],
            identity_for_transpose=identity_for_transpose,
        )


def PF_transpose_with_PE_int4byte(src, out, identity_for_transpose=None):
    """
    Perform int32 P-F Transpose with PE. Lower into 4 uint8 P-F transpose with reinterpret cast
    """
    # lower as 4 uint8 matmul
    assert src.dtype == out.dtype
    assert src.dtype == nl.int32
    p, f = src.shape
    src_copy = nl.copy(src)
    if identity_for_transpose is not None:
        assert identity_for_transpose.shape == (p, p), f"expect {(p, p)}"
        assert identity_for_transpose.dtype == nl.uint8
    else:
        identity_for_transpose_hbm = nl.shared_constant(
            np.identity(n=p, dtype=np.uint8),
            dtype=nl.uint8,
        )
        identity_for_transpose = nl.load(identity_for_transpose_hbm)
    src_reinterpreted = src_copy.view(nl.uint8)
    out_reinterpreted = out.view(nl.uint8)
    for i in nl.affine_range(4):
        out_psum = nl.ndarray((par_dim(f), p), dtype=nl.int32, buffer=nl.psum)
        i_p = nl.arange(p)[:, None]
        i_f = nl.arange(f)[None, :] * 4 + i
        out_psum[:, :] = nisa.nc_matmul(
            src_reinterpreted[i_p, i_f],
            identity_for_transpose,
            is_moving_onezero=True,
            is_transpose=True,
        )
        i_p = nl.arange(f)[:, None]
        i_f = nl.arange(p)[None, :]
        out_reinterpreted[i_p, i_f * 4 + i] = nl.copy(
            out_psum[i_p, i_f], dtype=nl.uint8
        )


def get_move_dtype(src):
    itemsize = src.itemsize
    assert itemsize <= 4, f"{src.dtype=} has itemsize > 4"
    if itemsize == 1:
        return nl.uint8
    elif itemsize == 2:
        return nl.bfloat16
    else:
        return nl.float32


def create_identity_for_transpose(src, size):
    identity_dtype = get_move_dtype(src)
    identity_for_transpose_hbm = nl.shared_constant(
        np.identity(n=size, dtype=np.uint8),
        dtype=identity_dtype,
    )
    identity_for_transpose = nl.load(identity_for_transpose_hbm)
    return identity_for_transpose


def PF_transpose_with_PE(src, out, identity_for_transpose=None, out_in_psum=False):
    """
    Perform P-F Transpose with PE.
    """
    p, f = src.shape
    assert p <= B_P_SIZE and f <= B_P_SIZE
    assert out.shape == (f, p), f"{src.shape=} {out.shape=}"

    if src.dtype == nl.int32:
        PF_transpose_with_PE_int4byte(src, out, identity_for_transpose)
        return

    move_dtype = get_move_dtype(src)
    if src.dtype != move_dtype:
        src_reinterpreted = src.view(move_dtype)
    else:
        src_reinterpreted = src
    if identity_for_transpose is None:
        identity_for_transpose_hbm = nl.shared_constant(
            np.identity(n=p, dtype=np.uint8),
            dtype=move_dtype,
        )
        identity_for_transpose = nl.load(identity_for_transpose_hbm)
    else:
        assert identity_for_transpose.shape == (p, p), f"expect {(p, p)}"
        assert (
            identity_for_transpose.dtype == move_dtype
        ), f"{identity_for_transpose.dtype=} expect {move_dtype=}"

    if out_in_psum:
        out_psum = out
    else:
        psum_dtype = nl.int32 if move_dtype == nl.uint8 else nl.float32
        out_psum = nl.ndarray(out.shape, dtype=psum_dtype, buffer=nl.psum)
    out_psum[:, :] = nisa.nc_matmul(
        src_reinterpreted,
        identity_for_transpose,
        is_moving_onezero=True,
        is_transpose=True,
    )
    if out_in_psum:
        assert out.dtype == nl.float32
    elif src.dtype == move_dtype:
        out[...] = nl.copy(out_psum, dtype=out.dtype)
    else:
        if src.dtype == out.dtype:
            out_reinterpreted = out.view(move_dtype)
            out_reinterpreted[...] = nl.copy(out_psum, dtype=move_dtype)
        else:
            out_tmp = nl.ndarray(out.shape, dtype=src.dtype)
            out_tmp_reinterpreted = out_tmp.view(move_dtype)
            out_tmp_reinterpreted[...] = nl.copy(out_psum, dtype=move_dtype)
            out[...] = nl.copy(out_tmp, dtype=out.dtype)


def broadcast_partition_with_PE(src, out, src_one_zero=False, out_in_psum=False):
    """
    Perform Partition Dimension Broadcast with PE rather than vector engine.
    """
    out_shape = out.shape
    assert (
        src.dtype != nl.int32
    ), f"{src.dtype=} may produce wrong results if input has negative values"
    assert len(src.shape) == 2 and len(out_shape) == 2
    assert src.shape[0] == 1 and src.shape[1] == out_shape[1]
    move_dtype = get_move_dtype(src)

    src_reinterpreted = src.view(move_dtype)
    ones = nl.ones((1, out_shape[0]), dtype=move_dtype)
    if out_in_psum:
        out_psum = out
    else:
        psum_dtype = nl.int32 if move_dtype == nl.uint8 else nl.float32
        out_psum = nl.ndarray(out_shape, dtype=psum_dtype, buffer=nl.psum)
    out_psum[:, :] = nisa.nc_matmul(
        ones,
        src_reinterpreted,
        is_stationary_onezero=True,
        is_moving_onezero=src_one_zero,
    )
    if out_in_psum:
        assert out.dtype == nl.float32
    elif src.dtype == move_dtype:
        out[...] = nl.copy(out_psum, dtype=out.dtype)
    else:
        if src.dtype == out.dtype:
            out_reinterpreted = out.view(move_dtype)
            out_reinterpreted[...] = nl.copy(out_psum, dtype=move_dtype)
        else:
            out_tmp = nl.ndarray(out.shape, dtype=src.dtype)
            out_tmp_reinterpreted = out_tmp.view(move_dtype)
            out_tmp_reinterpreted[...] = nl.copy(out_psum, dtype=move_dtype)
            out[...] = nl.copy(out_tmp, dtype=out.dtype)


def prepare_q_indices_range(
    tile_q_indices,
    INNER_Q_TILE_SIZE,
    loop_index=None,
    num_tiles_per_step=None,
):
    assert len(tile_q_indices.shape) == 2
    if loop_index is not None:
        q_indices_sbuf = load_indices_for_loop_step(
            tile_q_indices, loop_index, num_tiles_per_step
        )
    else:
        q_indices_sbuf = load_indices(tile_q_indices)
    tile_partition_size, num_tile_partitions, num_indices_per_tile = (
        q_indices_sbuf.shape
    )
    assert INNER_Q_TILE_SIZE <= B_P_SIZE
    index_partition_size = min(num_indices_per_tile, INNER_Q_TILE_SIZE)
    num_index_partitions = ceil_div(num_indices_per_tile, index_partition_size)
    transposed_q_indices = nl.ndarray(
        (
            par_dim(index_partition_size),
            num_index_partitions,
            num_tile_partitions * tile_partition_size,
        ),
        dtype=tile_q_indices.dtype,
    )
    transposed_q_indices_reshape = transposed_q_indices.reshape(
        (
            index_partition_size,
            num_index_partitions,
            num_tile_partitions,
            tile_partition_size,
        )
    )
    for i in range(num_tile_partitions):
        transform_to_vector_dge_layout(
            q_indices_sbuf[:, i],
            transposed_q_indices_reshape[:, :, i, :],
            index_partition_size,
        )
    return transposed_q_indices


def prepare_q_update_pred(last_tile_indices_sbuf, MAX_NUM_TILE):
    memset_tile_size = min(B_P_SIZE, MAX_NUM_TILE)
    update_pred = nl.ndarray((MAX_NUM_TILE, 1), dtype=nl.uint8, buffer=nl.hbm)
    for i in nl.affine_range(ceil_div(MAX_NUM_TILE, memset_tile_size)):
        i_p = nl.arange(memset_tile_size)[:, None] + i * memset_tile_size
        i_f = nl.arange(1)[None, :]
        nl.store(update_pred[i_p, i_f], 1, mask=(i_p < MAX_NUM_TILE))
    q_tile_size, num_tile = last_tile_indices_sbuf.shape
    for i in nl.affine_range(num_tile):
        i_p = nl.arange(q_tile_size)[:, None]
        i_f = nl.arange(1)[None, :]
        nl.store(
            update_pred[last_tile_indices_sbuf[i_p, i], i_f],
            0,
        )
    return update_pred


def load_and_broadcast_q_update_preds(q_update_pred, B_D_SIZE, loop_index=None):
    # q_update_pred has shape (1, MAX_NUM_TILE) for static loop case
    #                         (num_loop_step, loop_unroll_size) for dynamic loop case
    B_F_SIZE = B_FMAX_SIZE
    num_loop_steps, free_dim_size = q_update_pred.shape
    if loop_index is None:
        assert num_loop_steps == 1
    # PE broadcast can handle B_FMAX_SIZE=512 each time
    padded_free_dim_size = (
        free_dim_size
        if free_dim_size <= B_F_SIZE
        else pad_to_multiple(free_dim_size, B_F_SIZE)
    )
    out = nl.zeros((par_dim(1), padded_free_dim_size), dtype=nl.uint8)
    i_p = nl.arange(1)[:, None]
    i_f = nl.arange(free_dim_size)[None, :]
    if loop_index is None:
        out[i_p, i_f] = nl.load(
            q_update_pred[i_p, i_f],
            dtype=nl.uint8,
        )
    else:
        out[i_p, i_f] = nl.load(
            q_update_pred[loop_index[i_p, 0], i_f],
            dtype=nl.uint8,
        )
    out_broadcast = nl.ndarray(
        (par_dim(B_D_SIZE), padded_free_dim_size), dtype=nl.uint8
    )
    tile_size = min(B_F_SIZE, free_dim_size)
    num_tiles = padded_free_dim_size // tile_size
    out_broadcast_reshape = out_broadcast.reshape((B_D_SIZE, num_tiles, tile_size))
    for i in nl.affine_range(num_tiles):
        broadcast_partition_with_PE(
            src=out[:, nl.ds(i * tile_size, tile_size)],
            out=out_broadcast_reshape[:, i, :],
            src_one_zero=True,
            out_in_psum=False,
        )
    return out, out_broadcast


def prepare_decode_offsets(loop_index, num_tiles_unrolled):
    offsets = nisa.iota(nl.arange(num_tiles_unrolled)[None, :], dtype=nl.uint32)
    offsets_transposed = nl.ndarray((num_tiles_unrolled, 1), dtype=nl.uint32)
    PF_transpose_with_PE(
        nisa.tensor_tensor(offsets, loop_index * num_tiles_unrolled, nl.add),
        offsets_transposed,
        out_in_psum=False,
    )
    return offsets_transposed


def load_decode_query(
    query,
    tile_q_indices,
    q_indices_region,
    softmax_scale,
    batch_id,
    head_id,
    q_h_per_k_h,
    B_D_SIZE,
    kernel_dtype,
):
    if q_indices_region is None:
        load_size = tile_q_indices.shape[0]
    else:
        load_size = q_indices_region.shape[0]
    tile_size = min(load_size, B_P_SIZE)
    num_partitions = ceil_div(load_size, tile_size)
    query_scaled_transposed = nl.ndarray(
        (B_D_SIZE, tile_size * num_partitions, q_h_per_k_h),
        dtype=kernel_dtype,
    )
    identity_for_transpose = create_identity_for_transpose(
        query_scaled_transposed,
        tile_size,
    )
    q_indices_sbuf = nl.zeros(
        (par_dim(tile_size), num_partitions),
        dtype=tile_q_indices.dtype,
    )
    for load_idx in nl.affine_range(num_partitions):
        i_p = nl.arange(tile_size)[:, None]
        i_f = nl.arange(1)[None, :]
        if q_indices_region is None:
            q_indices_sbuf[i_p, i_f + load_idx] = nl.load(
                tile_q_indices[i_p + load_idx * tile_size, i_f],
                dtype=nl.uint32,
                mask=(i_p + load_idx * tile_size < load_size),
            )
        else:
            q_indices_sbuf[i_p, i_f + load_idx] = nl.load(
                tile_q_indices[q_indices_region[i_p + load_idx * tile_size, 0], i_f],
                dtype=nl.uint32,
                mask=(i_p + load_idx * tile_size < load_size),
            )
    for load_idx in nl.affine_range(num_partitions):
        for i_q_h in nl.affine_range(q_h_per_k_h):
            q_tile = nl.ndarray((tile_size, B_D_SIZE), dtype=query.dtype)
            i_p = nl.arange(tile_size)[:, None]
            i_f = nl.arange(B_D_SIZE)[None, :]
            q_tile[i_p, i_f] = nl.load(
                query[
                    batch_id,
                    head_id * q_h_per_k_h + i_q_h,
                    q_indices_sbuf[i_p, load_idx],
                    i_f,
                ]
            )
            q_tile_scaled = nl.multiply(q_tile, softmax_scale, dtype=kernel_dtype)
            q_transposed_psum = nl.ndarray(
                (par_dim(B_D_SIZE), tile_size),
                dtype=nl.float32,
                buffer=nl.psum,
            )
            PF_transpose_with_PE(
                q_tile_scaled,
                q_transposed_psum,
                identity_for_transpose=identity_for_transpose,
                out_in_psum=True,
            )
            query_scaled_transposed[
                :,
                nl.ds(load_idx * tile_size, tile_size),
                i_q_h,
            ] = nl.copy(q_transposed_psum, dtype=kernel_dtype)
    return query_scaled_transposed
