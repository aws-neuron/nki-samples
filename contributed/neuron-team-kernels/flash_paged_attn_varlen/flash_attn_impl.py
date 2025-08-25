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
from neuronxcc.nki.typing import scalar

from constants import NEG_INF, B_P_SIZE, B_FMAX_SIZE
from paged_cache import (
    prepare_kv_block_dim_tiling,
    transform_block_tables_for_indirect_load,
    load_k_tile_from_cache,
    load_v_tile_from_cache,
    transpose_k_cache_tile,
)
from flash_attn_core import (
    _flash_attention_core,
    _flash_attention_core_kq_matmul,
)
from utils import (
    ceil_div,
    pad_to_multiple,
    load_indices,
    load_indices_for_loop_step,
    transform_to_vector_dge_layout,
    broadcast_partition_with_PE,
    PF_transpose_with_PE,
    create_identity_for_transpose,
    IdentityStore,
)


def load_v_tile(v_hbm_tile, cur_v_tile, large_tile_idx, v_i, LARGE_TILE_SZ):
    B_D_SIZE = v_hbm_tile.shape[-1]
    cur_v_tile[:, nl.ds(v_i * B_D_SIZE, B_D_SIZE)] = nl.load(
        v_hbm_tile[
            nl.ds(large_tile_idx * LARGE_TILE_SZ + B_P_SIZE * v_i, B_P_SIZE),
            :,
        ],
        dtype=cur_v_tile.dtype,
    )


def allocate_prefill_accum_buffers(
    seqlen_q,
    INNER_Q_TILE_SIZE,
    q_h_per_k_h,
    B_D_SIZE,
    acc_type,
):
    # =============== Global Flash Attention accumulators ====================== #
    olm_buffer = nl.ndarray(
        (seqlen_q, q_h_per_k_h, B_D_SIZE + 2),
        dtype=acc_type,
        buffer=nl.hbm,
    )

    for i in nl.affine_range(ceil_div(seqlen_q, INNER_Q_TILE_SIZE)):
        i_x, i_y, i_z = nl.mgrid[
            :INNER_Q_TILE_SIZE, :q_h_per_k_h, : B_D_SIZE + 1
        ]
        i_x = i_x + i * INNER_Q_TILE_SIZE
        nl.store(
            dst=olm_buffer[i_x, i_y, i_z],
            value=0.0,
            mask=(i_x < seqlen_q),
        )
        i_x, i_y, i_z = nl.mgrid[
            :INNER_Q_TILE_SIZE,
            :q_h_per_k_h,
            B_D_SIZE + 1 : B_D_SIZE + 2,
        ]
        i_x = i_x + i * INNER_Q_TILE_SIZE
        nl.store(
            dst=olm_buffer[i_x, i_y, i_z],
            value=NEG_INF,
            mask=(i_x < seqlen_q),
        )

    # =============== Global Flash Attention accumulators END ================== #
    return (olm_buffer,)


def prefill_context_tokens(
    query,
    key_cache,
    value_cache,
    tile_q_indices,
    tile_block_tables,
    tile_masks,
    num_dynamic_loop_steps,
    olm_buffer,
    q_update_pred,
    kernel_dtype,
    acc_type,
    loop_unroll_factor,
    batch_id,
    head_id,
    k_h,
    q_h_per_k_h,
    softmax_scale,
    B_F_SIZE,
    B_D_SIZE,
):
    (
        INNER_Q_TILE_SIZE,
        MAX_NUM_TILE,
        n_small_in_large_q_tile,
        LARGE_KV_TILE_SIZE,
    ) = tile_masks.shape
    key_cache, value_cache, block_size_tiling_factor, block_size = (
        prepare_kv_block_dim_tiling(key_cache, value_cache, LARGE_KV_TILE_SIZE)
    )
    num_blocks_per_large_tile = LARGE_KV_TILE_SIZE // block_size
    olm_next_tile_sbuf = nl.zeros(
        (
            par_dim(INNER_Q_TILE_SIZE),
            n_small_in_large_q_tile,
            q_h_per_k_h,
            B_D_SIZE + 2,
        ),
        dtype=acc_type,
    )
    olm_next_tile_sbuf[:, :, :, B_D_SIZE + 1] = NEG_INF
    identity_store = IdentityStore(
        (kernel_dtype, B_P_SIZE),  # transpose k
        (kernel_dtype, INNER_Q_TILE_SIZE),  # transpose p
    )
    # prepare block tables
    tile_block_tables_sbuf = load_indices(tile_block_tables)
    tile_block_tables_transformed_sbuf = (
        transform_block_tables_for_indirect_load(
            tile_block_tables_sbuf,
            block_size_tiling_factor=block_size_tiling_factor,
            num_head=k_h,
            head_id=head_id,
            identity_for_transpose=None,
        )
    )
    num_loads = tile_block_tables_transformed_sbuf.shape[1]
    MAX_NUM_LOOP = MAX_NUM_TILE // loop_unroll_factor
    tile_block_tables_transformed_hbm = nl.ndarray(
        (MAX_NUM_LOOP, B_P_SIZE, num_loads, loop_unroll_factor),
        dtype=tile_block_tables.dtype,
        buffer=nl.hbm,
    )
    for i in nl.affine_range(MAX_NUM_LOOP):
        nl.store(
            tile_block_tables_transformed_hbm[i],
            tile_block_tables_transformed_sbuf[
                :, :, nl.ds(i * loop_unroll_factor, loop_unroll_factor)
            ],
        )

    block_tables_sbuf = nl.ndarray(
        tile_block_tables_transformed_hbm.shape[1:],
        dtype=tile_block_tables_transformed_hbm.dtype,
    )
    block_tables_sbuf[...] = nl.load(tile_block_tables_transformed_hbm[0])

    num_dynamic_loop_steps_sbuf = nl.load(num_dynamic_loop_steps)
    assert num_dynamic_loop_steps_sbuf.shape == (1, 1)
    if loop_unroll_factor > 1:
        num_unrolled_iota = nisa.iota(
            nl.arange(loop_unroll_factor)[None, :], dtype=nl.uint32
        )
    else:
        num_unrolled_iota = None

    MAX_NUM_BUFFER_ALLOWED = 2
    MULTI_BUFFER = min(MAX_NUM_BUFFER_ALLOWED, loop_unroll_factor)
    assert loop_unroll_factor % MULTI_BUFFER == 0
    # XXX: Work around a DMA skipping correctness issue:
    #      If nl.ndarray is used to allocate buffer for DMA skipping,
    #      kernel does not produce correct results.
    q_load_buffer = nl.zeros(
        (
            par_dim(INNER_Q_TILE_SIZE),
            MULTI_BUFFER,
            n_small_in_large_q_tile,
            q_h_per_k_h,
            B_D_SIZE,
        ),
        dtype=kernel_dtype,
        buffer=nl.sbuf,
    )
    num_loads = num_blocks_per_large_tile // B_P_SIZE
    k_load_buffer = nl.ndarray(
        (par_dim(B_P_SIZE), MULTI_BUFFER, num_loads, block_size * B_D_SIZE),
        dtype=key_cache.dtype,
    )
    v_load_buffer = nl.ndarray(
        (par_dim(B_P_SIZE), MULTI_BUFFER, num_loads * block_size * B_D_SIZE),
        dtype=value_cache.dtype,
    )
    olm_unrolled_sbuf = nl.ndarray(
        (
            par_dim(INNER_Q_TILE_SIZE),
            loop_unroll_factor + 1,
            n_small_in_large_q_tile,
            q_h_per_k_h,
            B_D_SIZE + 2,
        ),
        dtype=acc_type,
    )
    tile_masks = tile_masks.reshape(
        (
            INNER_Q_TILE_SIZE,
            MAX_NUM_TILE // loop_unroll_factor,
            loop_unroll_factor,
            n_small_in_large_q_tile,
            LARGE_KV_TILE_SIZE,
        )
    )
    mask_buffer = nl.ndarray(
        (
            par_dim(INNER_Q_TILE_SIZE),
            MULTI_BUFFER,
            n_small_in_large_q_tile,
            LARGE_KV_TILE_SIZE,
        ),
        dtype=tile_masks.dtype,
    )
    tile_q_indices_sbuf = nl.ndarray(
        (
            par_dim(INNER_Q_TILE_SIZE),
            n_small_in_large_q_tile,
            loop_unroll_factor,
        ),
        dtype=tile_q_indices.dtype,
    )
    loop_index = nl.zeros((1, 1), dtype=np.int32)
    prepare_q_indices_range(
        tile_q_indices,
        loop_index,
        loop_unroll_factor,
        INNER_Q_TILE_SIZE,
        tile_q_indices_sbuf,
        identity_for_transpose=None,
        num_tiles_iota=num_unrolled_iota,
    )
    mask_buffer[:, 0, :, :] = nl.load(tile_masks[:, 0, 0, :, :])
    q_update_pred_sbuf = nl.ndarray((loop_unroll_factor, 1), dtype=nl.uint8)
    q_update_pred_sbuf[...] = nl.load(q_update_pred[0])
    load_k_tile_from_cache(
        key_cache=key_cache,
        block_tables=block_tables_sbuf,
        large_k_tile_idx=0,
        num_blocks_per_large_tile=num_blocks_per_large_tile,
        block_size=block_size,
        B_D_SIZE=B_D_SIZE,
        k_load_buffer=k_load_buffer,
    )
    for _ in range(scalar(num_dynamic_loop_steps_sbuf)):
        olm_unrolled_sbuf[:, nl.ds(1, loop_unroll_factor)] = 0
        olm_unrolled_sbuf[:, :, :, :, B_D_SIZE + 1] = NEG_INF
        olm_unrolled_sbuf[:, 0] = nl.copy(olm_next_tile_sbuf)
        q_update_pred_broadcast = transpose_broadcast_q_pred(
            q_update_pred_sbuf, INNER_Q_TILE_SIZE
        )
        prefill_prior(
            loop_index=loop_index,
            num_tiles_unrolled=loop_unroll_factor,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            q_load_buffer=q_load_buffer,
            k_load_buffer=k_load_buffer,
            v_load_buffer=v_load_buffer,
            mask_buffer=mask_buffer,
            olm_buffer_hbm=olm_buffer,
            olm_unrolled_sbuf=olm_unrolled_sbuf,
            tile_q_indices_sbuf=tile_q_indices_sbuf,
            block_tables_sbuf=block_tables_sbuf,
            tile_masks=tile_masks,
            q_update_pred_broadcast=q_update_pred_broadcast,
            num_blocks_per_large_tile=num_blocks_per_large_tile,
            block_size=block_size,
            kernel_dtype=kernel_dtype,
            acc_type=acc_type,
            identity_store=identity_store,
            batch_id=batch_id,
            head_id=head_id,
            q_h_per_k_h=q_h_per_k_h,
            softmax_scale=softmax_scale,
            n_small_in_large_q_tile=n_small_in_large_q_tile,
            INNER_Q_TILE_SIZE=INNER_Q_TILE_SIZE,
            B_F_SIZE=B_F_SIZE,
            B_D_SIZE=B_D_SIZE,
        )
        # update loop_index
        loop_index[...] = nl.add(loop_index[...], 1)
        block_tables_sbuf[...] = nl.load(
            tile_block_tables_transformed_hbm[loop_index[0, 0]],
            mode=oob_mode.skip,
        )
        prepare_q_indices_range(
            tile_q_indices,
            loop_index,
            loop_unroll_factor,
            INNER_Q_TILE_SIZE,
            tile_q_indices_sbuf,
            identity_for_transpose=None,
            num_tiles_iota=num_unrolled_iota,
        )
        olm_next_tile_sbuf[...] = nl.copy(
            olm_unrolled_sbuf[:, loop_unroll_factor]
        )

        mask_buffer[:, 0, :, :] = nl.load(
            tile_masks[:, loop_index[0, 0], 0, :, :],
            mode=oob_mode.skip,
        )
        q_update_pred_sbuf[...] = nl.load(
            q_update_pred[loop_index[0, 0]],
            mode=oob_mode.skip,
        )
        load_k_tile_from_cache(
            key_cache=key_cache,
            block_tables=block_tables_sbuf,
            large_k_tile_idx=0,
            num_blocks_per_large_tile=num_blocks_per_large_tile,
            block_size=block_size,
            B_D_SIZE=B_D_SIZE,
            k_load_buffer=k_load_buffer,
        )


def prepare_q_indices_range(
    tile_q_indices,
    loop_index,
    num_tiles_per_step,
    INNER_Q_TILE_SIZE,
    tile_q_indices_sbuf,
    identity_for_transpose,
    num_tiles_iota,
):
    assert len(tile_q_indices.shape) == 2
    q_indices_sbuf = load_indices_for_loop_step(
        tile_q_indices,
        loop_index,
        num_tiles_per_step,
        partition_size_iota=num_tiles_iota,
    )
    tile_partition_size, num_tile_partitions, num_indices_per_tile = (
        q_indices_sbuf.shape
    )
    assert INNER_Q_TILE_SIZE <= B_P_SIZE
    index_partition_size = min(num_indices_per_tile, INNER_Q_TILE_SIZE)
    num_index_partitions = ceil_div(num_indices_per_tile, index_partition_size)
    transposed_q_indices_reshape = tile_q_indices_sbuf.reshape(
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
            identity_for_transpose=identity_for_transpose,
        )


def prefill_prior(
    loop_index,
    num_tiles_unrolled,
    query,
    key_cache,
    value_cache,
    q_load_buffer,
    k_load_buffer,
    v_load_buffer,
    mask_buffer,
    olm_buffer_hbm,
    olm_unrolled_sbuf,
    tile_q_indices_sbuf,
    block_tables_sbuf,
    tile_masks,
    q_update_pred_broadcast,
    num_blocks_per_large_tile,
    block_size,
    kernel_dtype,
    acc_type,
    identity_store,
    batch_id,
    head_id,
    q_h_per_k_h,
    softmax_scale,
    n_small_in_large_q_tile,
    INNER_Q_TILE_SIZE,
    B_F_SIZE,
    B_D_SIZE,
):
    """
    Handles `num_tiles_unrolled` tiles of attention between Q and context
    tokens by having a static sequential loop over the tiles.

    This function is also used as loop body of dynamic-loop kernel
    """
    MULTI_BUFFER = k_load_buffer.shape[1]
    identity_k, identity_p = identity_store.get(
        (kernel_dtype, B_P_SIZE),
        (kernel_dtype, INNER_Q_TILE_SIZE),
    )
    is_nc_gen2 = nisa.get_nc_version() == nisa.nc_version.gen2
    q_sbuf_tile_transposed = nl.ndarray(
        (
            par_dim(B_D_SIZE),
            MULTI_BUFFER,
            q_h_per_k_h,
            n_small_in_large_q_tile,
            INNER_Q_TILE_SIZE,
        ),
        dtype=query.dtype,
        buffer=nl.sbuf,
    )
    LARGE_KV_TILE_SIZE = tile_masks.shape[-1]
    transposed_k_tile = nl.ndarray(
        (par_dim(B_D_SIZE), MULTI_BUFFER, LARGE_KV_TILE_SIZE),
        dtype=kernel_dtype,
    )
    transpose_k_cache_tile(
        k_load_buffer=k_load_buffer,
        transposed_k_tile=transposed_k_tile,
        large_k_tile_idx=0,
        num_blocks_per_large_tile=num_blocks_per_large_tile,
        block_size=block_size,
        B_D_SIZE=B_D_SIZE,
        kernel_dtype=kernel_dtype,
        identity_for_transpose=identity_k,
    )
    for local_tile_idx in nl.sequential_range(num_tiles_unrolled):
        # load q
        for small_q_idx in nl.affine_range(n_small_in_large_q_tile):
            for i_q_h in nl.affine_range(q_h_per_k_h):
                i_p = nl.arange(INNER_Q_TILE_SIZE)[:, None]
                i_f = nl.arange(B_D_SIZE)[None, :]
                q_load_buffer[
                    i_p, local_tile_idx % MULTI_BUFFER, small_q_idx, i_q_h, i_f
                ] = nl.load(
                    query[
                        batch_id,
                        head_id * q_h_per_k_h + i_q_h,
                        tile_q_indices_sbuf[i_p, small_q_idx, local_tile_idx],
                        i_f,
                    ],
                    mode=oob_mode.skip,
                )
        for small_q_idx in nl.affine_range(n_small_in_large_q_tile):
            for i_q_h in nl.affine_range(q_h_per_k_h):
                q_t_psum = nl.ndarray(
                    (B_D_SIZE, INNER_Q_TILE_SIZE),
                    dtype=nl.float32 if is_nc_gen2 else query.dtype,
                    buffer=nl.psum,
                )
                PF_transpose_with_PE(
                    q_load_buffer[
                        :, local_tile_idx % MULTI_BUFFER, small_q_idx, i_q_h, :
                    ],
                    q_t_psum,
                    identity_for_transpose=identity_p,
                    out_in_psum=True,
                )
                q_sbuf_tile_transposed[
                    :, local_tile_idx % MULTI_BUFFER, i_q_h, small_q_idx, :
                ] = nl.multiply(
                    q_t_psum,
                    softmax_scale,
                    dtype=kernel_dtype,
                )
        cur_v_tile = load_v_tile_from_cache(
            value_cache=value_cache,
            block_tables=block_tables_sbuf,
            large_k_tile_idx=local_tile_idx,
            num_blocks_per_large_tile=num_blocks_per_large_tile,
            block_size=block_size,
            B_D_SIZE=B_D_SIZE,
            kernel_dtype=kernel_dtype,
            v_load_buffer=v_load_buffer,
        )

        def prefetch_and_transpose_k():
            load_k_tile_from_cache(
                key_cache=key_cache,
                block_tables=block_tables_sbuf,
                large_k_tile_idx=local_tile_idx + 1,
                num_blocks_per_large_tile=num_blocks_per_large_tile,
                block_size=block_size,
                B_D_SIZE=B_D_SIZE,
                k_load_buffer=k_load_buffer,
            )
            mask_buffer[:, (local_tile_idx + 1) % MULTI_BUFFER, :, :] = nl.load(
                tile_masks[:, loop_index[0, 0], local_tile_idx + 1, :, :]
            )
            # start transpose next k
            transpose_k_cache_tile(
                k_load_buffer=k_load_buffer,
                transposed_k_tile=transposed_k_tile,
                large_k_tile_idx=local_tile_idx + 1,
                num_blocks_per_large_tile=num_blocks_per_large_tile,
                block_size=block_size,
                B_D_SIZE=B_D_SIZE,
                kernel_dtype=kernel_dtype,
                identity_for_transpose=identity_k,
            )

        if MULTI_BUFFER > 1 and local_tile_idx + 1 < num_tiles_unrolled:
            prefetch_and_transpose_k()

        # perform compute
        for small_q_idx in nl.affine_range(n_small_in_large_q_tile):
            for i_q_h in nl.affine_range(q_h_per_k_h):
                q_tile = q_sbuf_tile_transposed[
                    :, local_tile_idx % MULTI_BUFFER, i_q_h, small_q_idx
                ]

                _flash_attention_core(
                    q_local_tile=q_tile,
                    k=transposed_k_tile[:, local_tile_idx % MULTI_BUFFER],
                    v=cur_v_tile,
                    olm_buffer=olm_unrolled_sbuf[
                        :, local_tile_idx, small_q_idx, i_q_h
                    ],
                    kernel_dtype=kernel_dtype,
                    acc_type=acc_type,
                    tile_mask=mask_buffer[
                        :, local_tile_idx % MULTI_BUFFER, small_q_idx
                    ],
                    use_causal_mask=False,
                    q_tile_idx=None,
                    Q_TILE_SIZE=INNER_Q_TILE_SIZE,
                    LARGE_KV_TILE_SIZE=LARGE_KV_TILE_SIZE,
                    B_F_SIZE=B_F_SIZE,
                    B_D_SIZE=B_D_SIZE,
                )
        nisa.tensor_copy_predicated(
            src=olm_unrolled_sbuf[:, local_tile_idx],
            dst=olm_unrolled_sbuf[:, local_tile_idx + 1],
            predicate=q_update_pred_broadcast[:, local_tile_idx],
        )

        if MULTI_BUFFER == 1 and local_tile_idx + 1 < num_tiles_unrolled:
            prefetch_and_transpose_k()

    for local_tile_idx in nl.affine_range(num_tiles_unrolled):
        # write out aggregation buffer
        for small_q_idx in nl.affine_range(n_small_in_large_q_tile):
            i_p = nl.arange(INNER_Q_TILE_SIZE)[:, None, None]
            i_f_h = nl.arange(q_h_per_k_h)[None, :, None]
            i_f_d = nl.arange(B_D_SIZE + 2)[None, None, :]
            nl.store(
                olm_buffer_hbm[
                    tile_q_indices_sbuf[i_p, small_q_idx, local_tile_idx],
                    i_f_h,
                    i_f_d,
                ],
                olm_unrolled_sbuf[
                    i_p, local_tile_idx, small_q_idx, i_f_h, i_f_d
                ],
                mode=oob_mode.skip,
            )


def active_and_epilogue(
    o,
    query,
    key,
    value,
    active_mask,
    softmax_scale,
    olm_buffer,
    ACTIVE_Q_TILE_SIZE,
    seqlen_q,
    batch_id,
    head_id,
    q_h_per_k_h,
    kernel_dtype,
    acc_type,
    B_F_SIZE,
    B_D_SIZE,
    skip_active,
):
    # -------- Load l, m, o back to SBUF from HBM ------------ #
    num_active_tiles = seqlen_q // ACTIVE_Q_TILE_SIZE
    assert seqlen_q % ACTIVE_Q_TILE_SIZE == 0

    olm_buffer_sbuf = nl.ndarray(
        (
            par_dim(ACTIVE_Q_TILE_SIZE),
            num_active_tiles,
            q_h_per_k_h,
            B_D_SIZE + 2,
        ),
        dtype=acc_type,
    )

    for i in nl.affine_range(num_active_tiles):
        olm_buffer_sbuf[:, i] = nl.load(
            olm_buffer[nl.ds(i * ACTIVE_Q_TILE_SIZE, ACTIVE_Q_TILE_SIZE)]
        )

    # compute attention between input query, key and value
    if not skip_active and key is not None and value is not None:
        B_F_SIZE = min(seqlen_q, B_F_SIZE)
        LARGE_KV_TILE_SIZE = seqlen_q
        cur_k_tile = nl.ndarray(
            (par_dim(B_D_SIZE), LARGE_KV_TILE_SIZE),
            dtype=kernel_dtype,
        )
        cur_k_tile[:, :] = nl.load(
            key[batch_id, head_id, :, :],
            dtype=cur_k_tile.dtype,
        )
        cur_v_tile = nl.ndarray(
            (par_dim(B_P_SIZE), LARGE_KV_TILE_SIZE // B_P_SIZE * B_D_SIZE),
            dtype=kernel_dtype,
        )
        v_hbm_tile = value[batch_id, head_id]
        # load at granularity of B_P_SIZE
        for v_i in nl.affine_range(LARGE_KV_TILE_SIZE // B_P_SIZE):
            load_v_tile(
                v_hbm_tile=v_hbm_tile,
                cur_v_tile=cur_v_tile,
                large_tile_idx=0,
                v_i=v_i,
                LARGE_TILE_SZ=LARGE_KV_TILE_SIZE,
            )

        (identity_for_transpose_p,) = create_identity_for_transpose(
            kernel_dtype,
            ACTIVE_Q_TILE_SIZE,
        )
        for i in nl.affine_range(num_active_tiles):
            cur_mask = nl.load(
                active_mask[
                    nl.ds(i * ACTIVE_Q_TILE_SIZE, ACTIVE_Q_TILE_SIZE),
                    nl.ds(0, LARGE_KV_TILE_SIZE),
                ],
                dtype=active_mask.dtype,
            )
            for i_q_h in nl.affine_range(q_h_per_k_h):
                q_hbm_tile = query[
                    batch_id,
                    head_id * q_h_per_k_h + i_q_h,
                ]
                q_sbuf_tile = nl.ndarray(
                    (ACTIVE_Q_TILE_SIZE, B_D_SIZE),
                    dtype=query.dtype,
                )
                q_sbuf_tile[:, :] = nl.load(
                    q_hbm_tile[
                        nl.ds(i * ACTIVE_Q_TILE_SIZE, ACTIVE_Q_TILE_SIZE),
                        :,
                    ],
                )  # load (d, 128) tile in SBUF
                q_tile_scaled = nl.ndarray(
                    (ACTIVE_Q_TILE_SIZE, B_D_SIZE),
                    dtype=kernel_dtype,
                )
                q_tile_scaled[:, :] = nl.multiply(
                    q_sbuf_tile,
                    softmax_scale,
                    dtype=kernel_dtype,
                )
                q_tile = nl.ndarray(
                    (B_D_SIZE, ACTIVE_Q_TILE_SIZE),
                    dtype=kernel_dtype,
                )
                PF_transpose_with_PE(
                    q_tile_scaled,
                    q_tile,
                    identity_for_transpose=identity_for_transpose_p,
                    out_in_psum=False,
                )
                _flash_attention_core(
                    q_local_tile=q_tile,
                    k=cur_k_tile,
                    v=cur_v_tile,
                    olm_buffer=olm_buffer_sbuf[:, i, i_q_h],
                    kernel_dtype=kernel_dtype,
                    acc_type=acc_type,
                    tile_mask=cur_mask,
                    use_causal_mask=True,
                    q_tile_idx=i,
                    Q_TILE_SIZE=ACTIVE_Q_TILE_SIZE,
                    LARGE_KV_TILE_SIZE=LARGE_KV_TILE_SIZE,
                    B_F_SIZE=B_F_SIZE,
                    B_D_SIZE=B_D_SIZE,
                )

    # -------- write output to buffer on HBM ------------ #
    for i in nl.affine_range(num_active_tiles):
        out = nl.ndarray(
            (par_dim(ACTIVE_Q_TILE_SIZE), q_h_per_k_h, B_D_SIZE),
            dtype=kernel_dtype,
        )
        out[...] = nl.multiply(
            olm_buffer_sbuf[:, i, :, nl.ds(0, B_D_SIZE)],
            # XXX: l is 0 in padded tokens
            1.0 / olm_buffer_sbuf[:, i, :, nl.ds(B_D_SIZE, 1)],
            dtype=kernel_dtype,
        )
        for i_q_h in nl.affine_range(q_h_per_k_h):
            nl.store(
                o[
                    batch_id,
                    head_id * q_h_per_k_h + i_q_h,
                    nl.ds(i * ACTIVE_Q_TILE_SIZE, ACTIVE_Q_TILE_SIZE),
                    :,
                ],
                out[:, i_q_h, :],
            )


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


def load_and_broadcast_q_update_preds(q_update_pred, loop_index, B_D_SIZE):
    B_F_SIZE = B_FMAX_SIZE
    _, free_dim_size = q_update_pred.shape
    padded_free_dim_size = (
        free_dim_size
        if free_dim_size <= B_F_SIZE
        else pad_to_multiple(free_dim_size, B_F_SIZE)
    )
    out = nl.zeros((par_dim(1), padded_free_dim_size), dtype=nl.uint8)
    i_p = nl.arange(1)[:, None]
    i_f = nl.arange(free_dim_size)[None, :]
    out[i_p, i_f] = nl.load(
        q_update_pred[loop_index[i_p, 0], i_f],
        dtype=nl.uint8,
    )
    if nisa.get_nc_version() == nisa.nc_version.gen3:
        out_broadcast = nl.broadcast_to(
            out, shape=(B_D_SIZE, padded_free_dim_size)
        )
    else:
        out_broadcast = nl.ndarray(
            (par_dim(B_D_SIZE), padded_free_dim_size),
            dtype=nl.uint8,
        )
        tile_size = min(B_F_SIZE, free_dim_size)
        num_tiles = padded_free_dim_size // tile_size
        out_broadcast_reshape = out_broadcast.reshape(
            (B_D_SIZE, num_tiles, tile_size)
        )
        for i in nl.affine_range(num_tiles):
            broadcast_partition_with_PE(
                src=out[:, nl.ds(i * tile_size, tile_size)],
                out=out_broadcast_reshape[:, i, :],
                src_one_zero=True,
                out_in_psum=False,
            )
    return out, out_broadcast


def allocate_decode_accum_buffers(
    MAX_NUM_TILE,
    q_h_per_k_h,
    B_D_SIZE,
    acc_type,
):
    # =============== Global Flash Attention accumulators ====================== #
    olm_buffer = nl.ndarray(
        (par_dim(MAX_NUM_TILE), q_h_per_k_h, B_D_SIZE + 2),
        dtype=acc_type,
        buffer=nl.hbm,
    )
    # =============== Global Flash Attention accumulators END ================== #
    return (olm_buffer,)


def transpose_broadcast_q_pred(q_update_pred_sbuf, partition_size):
    assert partition_size <= B_P_SIZE
    num_tiles_unrolled = q_update_pred_sbuf.shape[0]
    q_update_pred_broadcast = nl.ndarray(
        (num_tiles_unrolled, partition_size),
        dtype=nl.bfloat16,
    )
    q_update_pred_broadcast[:, :] = nl.copy(
        q_update_pred_sbuf.broadcast_to((num_tiles_unrolled, partition_size)),
        dtype=nl.bfloat16,
    )
    if nisa.get_nc_version() == nisa.nc_version.gen2:
        q_update_pred_broadcast_transposed = nl.ndarray(
            (partition_size, num_tiles_unrolled),
            dtype=nl.uint8,
        )
        PF_transpose_with_PE(
            src=q_update_pred_broadcast,
            out=q_update_pred_broadcast_transposed,
            out_in_psum=False,
        )
    else:
        q_update_pred_transposed = nl.ndarray(
            (partition_size, num_tiles_unrolled),
            dtype=nl.bfloat16,
            buffer=nl.psum,
        )
        q_update_pred_transposed[...] = nisa.nc_transpose(
            q_update_pred_broadcast,
            engine=nisa.tensor_engine,
        )
        q_update_pred_broadcast_transposed = nl.copy(
            q_update_pred_transposed,
            dtype=nl.uint8,
        )
    return q_update_pred_broadcast_transposed


def decode_context_tokens(
    query,
    key_cache,
    value_cache,
    tile_q_indices,
    tile_masks,
    tile_block_tables,
    num_dynamic_loop_steps,
    olm_buffer,
    q_update_pred,
    kernel_dtype,
    acc_type,
    loop_unroll_factor,
    batch_id,
    head_id,
    k_h,
    q_h_per_k_h,
    softmax_scale,
    B_D_SIZE,
):
    INNER_KV_TILE_SIZE, _, N_INNER_KV_TILE = tile_masks.shape
    LARGE_KV_TILE_SIZE = INNER_KV_TILE_SIZE * N_INNER_KV_TILE
    key_cache, value_cache, block_size_tiling_factor, block_size = (
        prepare_kv_block_dim_tiling(key_cache, value_cache, LARGE_KV_TILE_SIZE)
    )
    num_blocks_per_large_tile = LARGE_KV_TILE_SIZE // block_size
    # prepare block tables
    tile_block_tables_sbuf = load_indices(tile_block_tables)
    tile_block_tables_transformed_sbuf = (
        transform_block_tables_for_indirect_load(
            tile_block_tables_sbuf,
            block_size_tiling_factor=block_size_tiling_factor,
            num_head=k_h,
            head_id=head_id,
            identity_for_transpose=None,
        )
    )
    MAX_NUM_TILE = tile_block_tables.shape[0]
    num_loads = tile_block_tables_transformed_sbuf.shape[1]
    MAX_NUM_LOOP = MAX_NUM_TILE // loop_unroll_factor
    tile_block_tables_transformed_hbm = nl.ndarray(
        (MAX_NUM_LOOP, B_P_SIZE, num_loads, loop_unroll_factor),
        dtype=tile_block_tables.dtype,
        buffer=nl.hbm,
    )
    for i in nl.affine_range(MAX_NUM_LOOP):
        nl.store(
            tile_block_tables_transformed_hbm[i],
            tile_block_tables_transformed_sbuf[
                :, :, nl.ds(i * loop_unroll_factor, loop_unroll_factor)
            ],
        )

    olm_next_tile_sbuf = nl.zeros(
        (par_dim(B_D_SIZE), 3, q_h_per_k_h),
        dtype=acc_type,
    )
    olm_next_tile_sbuf[:, nl.ds(2, 1), :] = NEG_INF

    identity_store = IdentityStore(
        (acc_type, B_P_SIZE),  # transpose max cascaded step 1
        (acc_type, q_h_per_k_h),  # transpose max casesded step 2
        (kernel_dtype, B_P_SIZE),  # transpose k
        (kernel_dtype, loop_unroll_factor),  # transpose q
        (acc_type, B_D_SIZE),  # transpose o
    )

    tile_q_indices = tile_q_indices.reshape(
        (MAX_NUM_TILE // loop_unroll_factor, loop_unroll_factor, 1)
    )
    q_indices = nl.ndarray((loop_unroll_factor, 1), dtype=tile_q_indices.dtype)
    q_indices[...] = nl.load(tile_q_indices[0])
    block_tables_sbuf = nl.ndarray(
        tile_block_tables_transformed_hbm.shape[1:],
        dtype=tile_block_tables_transformed_hbm.dtype,
    )
    block_tables_sbuf[...] = nl.load(tile_block_tables_transformed_hbm[0])
    q_update_pred_sbuf = nl.ndarray((loop_unroll_factor, 1), dtype=nl.uint8)
    q_update_pred_sbuf[...] = nl.load(q_update_pred[0])
    num_k_tiles = tile_masks.shape[2]
    assert num_k_tiles == num_loads * block_size
    assert tile_masks.shape[0] == B_P_SIZE
    tile_masks = tile_masks.reshape(
        (
            B_P_SIZE,
            MAX_NUM_TILE // loop_unroll_factor,
            loop_unroll_factor,
            num_k_tiles,
        )
    )
    tile_mask_sbuf = nl.ndarray(
        (par_dim(B_P_SIZE), loop_unroll_factor, num_k_tiles),
        dtype=tile_masks.dtype,
    )
    tile_mask_sbuf[...] = nl.load(tile_masks[:, 0, :, :])

    num_dynamic_loop_steps_sbuf = nl.load(num_dynamic_loop_steps)
    assert num_dynamic_loop_steps_sbuf.shape == (1, 1)
    olm_buffer_reshaped = olm_buffer.reshape(
        (MAX_NUM_TILE // loop_unroll_factor, loop_unroll_factor)
        + olm_buffer.shape[1:]
    )

    (identity_o, identity_q) = identity_store.get(
        (acc_type, B_D_SIZE),
        (kernel_dtype, loop_unroll_factor),
    )
    query_sbuf = nl.ndarray(
        (B_D_SIZE, q_indices.shape[0], q_h_per_k_h),
        dtype=kernel_dtype,
    )
    load_decode_query(
        query_sbuf=query_sbuf,
        q_indices=q_indices,
        query=query,
        softmax_scale=softmax_scale,
        batch_id=batch_id,
        head_id=head_id,
        q_h_per_k_h=q_h_per_k_h,
        B_D_SIZE=B_D_SIZE,
        kernel_dtype=kernel_dtype,
        identity_for_transpose=identity_q,
    )
    MAX_NUM_BUFFER_ALLOWED = 2
    MULTI_BUFFER = min(MAX_NUM_BUFFER_ALLOWED, loop_unroll_factor)
    assert loop_unroll_factor % MULTI_BUFFER == 0

    # load first iteration of key cache
    k_load_buffer = nl.ndarray(
        (B_P_SIZE, MULTI_BUFFER, num_loads, block_size * B_D_SIZE),
        dtype=key_cache.dtype,
    )
    for load_idx in nl.affine_range(num_loads):
        i_p = nl.arange(B_P_SIZE)[:, None]
        i_f = nl.arange(block_size * B_D_SIZE)[None, :]
        k_load_buffer[i_p, 0, load_idx, i_f] = nl.load(
            key_cache[block_tables_sbuf[i_p, load_idx, 0], i_f]
        )

    identity_for_transpose_lm = nl.ones((1, 1), dtype=acc_type)
    # load all q and multiply scale

    loop_index = nl.zeros((1, 1), dtype=np.int32)
    for _ in range(scalar(num_dynamic_loop_steps_sbuf)):
        q_update_pred_broadcast = transpose_broadcast_q_pred(
            q_update_pred_sbuf, B_D_SIZE
        )
        o_buffer_sbuf = nl.zeros(
            (par_dim(B_D_SIZE), loop_unroll_factor + 1, q_h_per_k_h),
            dtype=acc_type,
        )
        m_buffer_sbuf = nl.full(
            (par_dim(1), loop_unroll_factor + 1, q_h_per_k_h),
            NEG_INF,
            dtype=acc_type,
        )
        l_buffer_sbuf = nl.zeros(
            (par_dim(1), loop_unroll_factor + 1, q_h_per_k_h),
            dtype=acc_type,
        )

        current_index = nl.copy(loop_index)

        # update loop_index
        loop_index[...] = nl.add(loop_index[...], 1)

        decode_prior(
            num_tiles_unrolled=loop_unroll_factor,
            query_sbuf=query_sbuf,
            key_cache=key_cache,
            value_cache=value_cache,
            olm_next_tile_sbuf=olm_next_tile_sbuf,
            o_buffer_sbuf=o_buffer_sbuf,
            l_buffer_sbuf=l_buffer_sbuf,
            m_buffer_sbuf=m_buffer_sbuf,
            k_load_buffer=k_load_buffer,
            block_tables_sbuf=block_tables_sbuf,
            tile_mask_sbuf=tile_mask_sbuf,
            q_update_pred_broadcast=q_update_pred_broadcast,
            num_blocks_per_large_tile=num_blocks_per_large_tile,
            block_size=block_size,
            kernel_dtype=kernel_dtype,
            acc_type=acc_type,
            q_h_per_k_h=q_h_per_k_h,
            B_D_SIZE=B_D_SIZE,
            identity_store=identity_store,
        )

        q_indices[...] = nl.load(
            tile_q_indices[loop_index[0, 0]],
            mode=oob_mode.skip,
        )
        q_update_pred_sbuf[...] = nl.load(
            q_update_pred[loop_index[0, 0]],
            mode=oob_mode.skip,
        )
        block_tables_sbuf[...] = nl.load(
            tile_block_tables_transformed_hbm[loop_index[0, 0]],
            mode=oob_mode.skip,
        )
        i_p = nl.arange(B_P_SIZE)[:, None, None]
        i_q = nl.arange(loop_unroll_factor)[None, :, None]
        i_k = nl.arange(num_k_tiles)[None, None, :]
        tile_mask_sbuf[i_p, i_q, i_k] = nl.load(
            tile_masks[i_p, loop_index[0, 0], i_q, i_k], mode=oob_mode.skip
        )

        # store L, M, O in (seqlen, h, d) format
        olm_sbuf = nl.ndarray(
            (loop_unroll_factor, q_h_per_k_h, B_D_SIZE + 2),
            dtype=o_buffer_sbuf.dtype,
        )
        for i_q_h in nl.affine_range(q_h_per_k_h):
            o_tmp = nl.ndarray(
                (loop_unroll_factor, B_D_SIZE),
                dtype=acc_type,
                buffer=nl.psum,
            )
            PF_transpose_with_PE(
                src=o_buffer_sbuf[:, nl.ds(0, loop_unroll_factor), i_q_h],
                out=o_tmp,
                identity_for_transpose=identity_o,
                out_in_psum=True,
            )
            olm_sbuf[:, i_q_h, nl.ds(0, B_D_SIZE)] = nl.copy(o_tmp)
            l_tmp = nl.ndarray(
                (loop_unroll_factor, 1),
                dtype=acc_type,
                buffer=nl.psum,
            )
            PF_transpose_with_PE(
                src=l_buffer_sbuf[:, nl.ds(0, loop_unroll_factor), i_q_h],
                out=l_tmp,
                identity_for_transpose=identity_for_transpose_lm,
                out_in_psum=True,
            )
            olm_sbuf[:, i_q_h, nl.ds(B_D_SIZE, 1)] = nl.copy(l_tmp)
            m_tmp = nl.ndarray(
                (loop_unroll_factor, 1),
                dtype=acc_type,
                buffer=nl.psum,
            )
            PF_transpose_with_PE(
                src=m_buffer_sbuf[:, nl.ds(0, loop_unroll_factor), i_q_h],
                out=m_tmp,
                identity_for_transpose=identity_for_transpose_lm,
                out_in_psum=True,
            )
            olm_sbuf[:, i_q_h, nl.ds(B_D_SIZE + 1, 1)] = nl.copy(m_tmp)

        nl.store(olm_buffer_reshaped[current_index[0, 0]], olm_sbuf)
        load_decode_query(
            query_sbuf=query_sbuf,
            q_indices=q_indices,
            query=query,
            softmax_scale=softmax_scale,
            batch_id=batch_id,
            head_id=head_id,
            q_h_per_k_h=q_h_per_k_h,
            B_D_SIZE=B_D_SIZE,
            kernel_dtype=kernel_dtype,
            identity_for_transpose=identity_q,
        )
        for load_idx in nl.affine_range(num_loads):
            i_p = nl.arange(B_P_SIZE)[:, None]
            i_f = nl.arange(block_size * B_D_SIZE)[None, :]
            k_load_buffer[i_p, 0, load_idx, i_f] = nl.load(
                key_cache[block_tables_sbuf[i_p, load_idx, 0], i_f]
            )


def load_decode_query(
    query_sbuf,
    q_indices,
    query,
    softmax_scale,
    batch_id,
    head_id,
    q_h_per_k_h,
    B_D_SIZE,
    kernel_dtype,
    identity_for_transpose,
):
    load_size = q_indices.shape[0]
    is_nc_gen2 = nisa.get_nc_version() == nisa.nc_version.gen2
    for i_q_h in nl.affine_range(q_h_per_k_h):
        i_p = nl.arange(load_size)[:, None]
        i_f = nl.arange(B_D_SIZE)[None, :]
        q_tile = nl.ndarray((load_size, B_D_SIZE), dtype=query.dtype)
        q_tile[i_p, i_f] = nl.load(
            query[
                batch_id,
                head_id * q_h_per_k_h + i_q_h,
                q_indices[i_p, 0],
                i_f,
            ]
        )
        q_t_psum = nl.ndarray(
            (B_D_SIZE, load_size),
            dtype=nl.float32 if is_nc_gen2 else kernel_dtype,
            buffer=nl.psum,
        )
        PF_transpose_with_PE(
            q_tile,
            q_t_psum,
            identity_for_transpose=identity_for_transpose,
            out_in_psum=True,
        )
        query_sbuf[:, :, i_q_h] = nl.multiply(
            q_t_psum,
            softmax_scale,
            dtype=kernel_dtype,
        )


def decode_prior(
    num_tiles_unrolled,
    query_sbuf,
    key_cache,
    value_cache,
    olm_next_tile_sbuf,
    o_buffer_sbuf,
    l_buffer_sbuf,
    m_buffer_sbuf,
    k_load_buffer,
    block_tables_sbuf,
    tile_mask_sbuf,
    q_update_pred_broadcast,
    num_blocks_per_large_tile,
    block_size,
    kernel_dtype,
    acc_type,
    q_h_per_k_h,
    B_D_SIZE,
    identity_store,
):
    """
    Handles `num_tiles_unrolled` tiles of attention between Q and context tokens
    by having a static sequential loop over the tiles.

    This function is also used as loop body of dynamic-loop kernel
    """
    identity_m_step1, identity_m_step2, identity_k = identity_store.get(
        (acc_type, B_P_SIZE),
        (acc_type, q_h_per_k_h),
        (kernel_dtype, B_P_SIZE),
    )
    assert q_h_per_k_h <= B_P_SIZE

    num_loads = num_blocks_per_large_tile // B_P_SIZE

    sumexp_ones = nl.ones((par_dim(B_P_SIZE), 1), dtype=kernel_dtype)

    o_buffer_sbuf[:, 0, :] = nl.copy(olm_next_tile_sbuf[:, 0, :])
    l_buffer_sbuf[:, 0, :] = nl.copy(olm_next_tile_sbuf[nl.ds(0, 1), 1, :])
    m_buffer_sbuf[:, 0, :] = nl.copy(olm_next_tile_sbuf[nl.ds(0, 1), 2, :])

    MULTI_BUFFER = k_load_buffer.shape[1]
    v_load_buffer = nl.ndarray(
        (par_dim(B_P_SIZE), MULTI_BUFFER, num_loads * block_size * B_D_SIZE),
        dtype=kernel_dtype,
    )

    for local_tile_idx in nl.sequential_range(num_tiles_unrolled):
        _flash_attention_core_kq_matmul(
            q_local_tile=query_sbuf,
            o_buffer_sbuf=o_buffer_sbuf,
            l_buffer_sbuf=l_buffer_sbuf,
            m_buffer_sbuf=m_buffer_sbuf,
            kernel_dtype=kernel_dtype,
            acc_type=acc_type,
            q_update_pred_broadcast=q_update_pred_broadcast,
            tile_mask=tile_mask_sbuf,
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables_sbuf=block_tables_sbuf,
            large_tile_idx=local_tile_idx,
            num_blocks_per_large_tile=num_blocks_per_large_tile,
            block_size=block_size,
            k_load_buffer=k_load_buffer,
            v_load_buffer=v_load_buffer,
            sumexp_ones=sumexp_ones,
            identity_for_transpose_k=identity_k,
            identity_for_transpose_m_step1=identity_m_step1,
            identity_for_transpose_m_step2=identity_m_step2,
        )

    olm_next_tile_sbuf[:, 0, :] = nl.copy(
        o_buffer_sbuf[:, num_tiles_unrolled, :]
    )
    olm_next_tile_sbuf[nl.ds(0, 1), 1, :] = nl.copy(
        l_buffer_sbuf[:, num_tiles_unrolled, :]
    )
    olm_next_tile_sbuf[nl.ds(0, 1), 2, :] = nl.copy(
        m_buffer_sbuf[:, num_tiles_unrolled, :]
    )


def decode_gather_token_last_accum_tile(
    olm_buffer,
    last_tile_indices_sbuf,
    q_h_per_k_h,
    B_D_SIZE,
):
    acc_type = olm_buffer.dtype
    TILE_SIZE, NUM_TILES = last_tile_indices_sbuf.shape
    olm_buffer_sbuf = nl.ndarray(
        (par_dim(TILE_SIZE), NUM_TILES, q_h_per_k_h, B_D_SIZE + 2),
        dtype=acc_type,
    )
    for i in nl.affine_range(NUM_TILES):
        i_q = nl.arange(TILE_SIZE)[:, None, None]
        i_h = nl.arange(q_h_per_k_h)[None, :, None]
        i_d = nl.arange(B_D_SIZE + 2)[None, None, :]
        olm_buffer_sbuf[i_q, i, i_h, i_d] = nl.load(
            olm_buffer[last_tile_indices_sbuf[i_q, i], i_h, i_d]
        )
    return olm_buffer_sbuf
