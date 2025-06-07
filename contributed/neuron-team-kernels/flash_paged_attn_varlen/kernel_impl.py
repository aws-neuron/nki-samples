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
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc.nki.language import par_dim
from neuronxcc.nki.typing import scalar

from paged_cache import (
    prepare_kv_block_dim_tiling,
    transform_block_tables_for_indirect_load,
)
from flash_attn_helper import (
    prefill_prior_tokens,
    prefill_active_tokens_and_epilogue,
    decode_prior_tokens,
    decode_active_tokens_and_epilogue,
)
from utils import (
    B_P_SIZE,
    B_FMAX_SIZE,
    ceil_div,
    is_power_of_2,
    load_indices,
    load_indices_for_loop_step,
    prepare_q_indices_range,
    prepare_decode_offsets,
    prepare_q_update_pred,
    load_decode_query,
    load_and_broadcast_q_update_preds,
)


def check_input_shapes(
    query, key, value, key_cache, value_cache, tile_masks, active_mask, decode_mode
):
    if decode_mode:
        INNER_KV_TILE_SIZE, _, N_INNER_KV_TILE = tile_masks.shape
        LARGE_Q_TILE_SIZE = 1
        assert INNER_KV_TILE_SIZE == B_P_SIZE
        LARGE_KV_TILE_SIZE = INNER_KV_TILE_SIZE * N_INNER_KV_TILE
    else:
        _, LARGE_Q_TILE_SIZE, LARGE_KV_TILE_SIZE = tile_masks.shape
    b, h, seqlen_q, d = query.shape
    assert seqlen_q <= 8192, f"Large {seqlen_q=} consumes too much sbuf space"
    if seqlen_q <= B_P_SIZE:
        assert is_power_of_2(seqlen_q), f"{seqlen_q=} is expected to be power of 2"
    elif seqlen_q <= B_FMAX_SIZE:
        assert seqlen_q % B_P_SIZE == 0, f"{seqlen_q=} must be mulitple of {B_P_SIZE=}"
    else:
        assert (
            seqlen_q % B_FMAX_SIZE == 0
        ), f"{seqlen_q=} must be multiple of {B_FMAX_SIZE=}"
    assert (
        seqlen_q % LARGE_Q_TILE_SIZE == 0
    ), f"{seqlen_q=} must be multiple of {LARGE_Q_TILE_SIZE=}"
    assert b == 1, f"Batch size must be 1 for Ragged Tensor, got {b}"
    assert (
        d >= 16 and d <= 128 and is_power_of_2(d)
    ), f" we head_dim must be power of 2 in range [16, 128], got head dim {d}"
    num_blocks, k_h, block_size, _ = key_cache.shape
    assert tuple(key_cache.shape) == (
        num_blocks,
        k_h,
        block_size,
        d,
    ), f"{key_cache.shape=} mismatch!"
    assert tuple(value_cache.shape) == (
        num_blocks,
        k_h,
        block_size,
        d,
    ), f"{value_cache.shape=} mismatch!"
    assert key is None or tuple(key.shape) == (
        1,
        k_h,
        d,
        seqlen_q,
    ), f"key shape {key.shape} mismatch!"
    assert value is None or tuple(value.shape) == (
        1,
        k_h,
        seqlen_q,
        d,
    ), f"value shape {value.shape} mismatch!"
    assert tile_masks.dtype == nl.uint8, f"{tile_masks.dtype=} is expected to be uint8"
    assert (
        active_mask.dtype == nl.uint8
    ), f"{active_mask.dtype=} is expected to be uint8"

    INNER_Q_TILE_SIZE = min(B_P_SIZE, LARGE_Q_TILE_SIZE)
    assert LARGE_Q_TILE_SIZE % INNER_Q_TILE_SIZE == 0

    assert (
        nl.program_ndim() == 2
    ), f"Expect spmd grid with 2 dimensions, got {nl.program_ndim()} instead!"
    assert nl.num_programs(0) == 1
    assert nl.num_programs(1) == k_h
    return b, h, k_h, seqlen_q, d, LARGE_KV_TILE_SIZE, INNER_Q_TILE_SIZE


@nki.compiler.skip_middle_end_transformations
@nki.jit
def flash_paged_attn_blockspase_prefill_static(
    query,
    key,
    value,
    key_cache,
    value_cache,
    tile_q_indices,
    tile_block_tables,
    tile_masks,
    active_mask,
    softmax_scale=None,
    mixed_precision=True,
):
    b, h, k_h, seqlen_q, d, LARGE_KV_TILE_SIZE, INNER_Q_TILE_SIZE = check_input_shapes(
        query, key, value, key_cache, value_cache, tile_masks, active_mask, False
    )
    batch_id = nl.program_id(axis=0)  # equals 0
    head_id = nl.program_id(axis=1)
    kernel_dtype = nl.bfloat16 if mixed_precision else query.dtype
    acc_type = np.dtype(np.float32) if mixed_precision else kernel_dtype
    o = nl.ndarray((b, h, seqlen_q, d), dtype=query.dtype, buffer=nl.shared_hbm)
    softmax_scale = softmax_scale or (1.0 / (d**0.5))
    q_h_per_k_h = h // k_h
    B_D_SIZE = d

    key_cache, value_cache, block_size_tiling_factor, block_size = (
        prepare_kv_block_dim_tiling(key_cache, value_cache, LARGE_KV_TILE_SIZE)
    )
    num_blocks_per_large_tile = LARGE_KV_TILE_SIZE // block_size

    B_F_SIZE = B_FMAX_SIZE
    MAX_NUM_TILE, LARGE_Q_TILE_SIZE, LARGE_KV_TILE_SIZE = tile_masks.shape
    assert (
        LARGE_KV_TILE_SIZE % B_F_SIZE == 0
    ), f"Need LARGE_KV_TILE_SIZE ({LARGE_KV_TILE_SIZE=}) to be divisible by ({B_F_SIZE=})"
    n_small_in_large_q_tile = LARGE_Q_TILE_SIZE // INNER_Q_TILE_SIZE
    NEG_INF = -9984.0  # Magic number to replace -inf similar to what Tensorizer uses
    # =============== Global Flash Attention accumulators ====================== #
    o_buffer = nl.ndarray(
        (seqlen_q, q_h_per_k_h, B_D_SIZE),
        dtype=acc_type,
        buffer=nl.hbm,
    )
    m_buffer = nl.ndarray(
        (seqlen_q, q_h_per_k_h, 1),
        dtype=acc_type,
        buffer=nl.hbm,
    )
    # FIXME: rename l_buffer as sumexp_buffer
    l_buffer = nl.ndarray(
        (seqlen_q, q_h_per_k_h, 1),
        dtype=acc_type,
        buffer=nl.hbm,
    )

    for i in nl.affine_range(ceil_div(seqlen_q, INNER_Q_TILE_SIZE)):
        i_x, i_y, i_z = nl.mgrid[:INNER_Q_TILE_SIZE, :q_h_per_k_h, :B_D_SIZE]
        i_x = i_x + i * INNER_Q_TILE_SIZE
        nl.store(dst=o_buffer[i_x, i_y, i_z], value=0.0, mask=(i_x < seqlen_q))
        i_x, i_y, i_z = nl.mgrid[:INNER_Q_TILE_SIZE, :q_h_per_k_h, :1]
        i_x = i_x + i * INNER_Q_TILE_SIZE
        nl.store(dst=m_buffer[i_x, i_y, i_z], value=NEG_INF, mask=(i_x < seqlen_q))
        nl.store(dst=l_buffer[i_x, i_y, i_z], value=0, mask=(i_x < seqlen_q))

    # =============== Global Flash Attention accumulators END ================== #

    # transpose identity matrix on hbm
    identity_for_transpose_k_hbm = nl.shared_constant(
        np.identity(n=B_P_SIZE, dtype=np.uint8),
        dtype=kernel_dtype,
    )
    identity_for_transpose_p_hbm = nl.shared_constant(
        np.identity(n=INNER_Q_TILE_SIZE, dtype=np.uint8),
        dtype=kernel_dtype,
    )

    tile_q_indices_sbuf = prepare_q_indices_range(
        tile_q_indices,
        INNER_Q_TILE_SIZE,
    )
    block_tables_sbuf = load_indices(
        tile_block_tables,
    )
    block_tables_sbuf = transform_block_tables_for_indirect_load(
        block_tables_sbuf,
        block_size_tiling_factor=block_size_tiling_factor,
        num_head=k_h,
        head_id=head_id,
    )
    prefill_prior_tokens(
        num_tiles_unrolled=MAX_NUM_TILE,
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        m_buffer=m_buffer,
        l_buffer=l_buffer,
        o_buffer=o_buffer,
        tile_q_indices_sbuf=tile_q_indices_sbuf,
        cur_masks=None,
        tile_masks=tile_masks,
        block_tables_sbuf=block_tables_sbuf,
        num_blocks_per_large_tile=num_blocks_per_large_tile,
        block_size=block_size,
        kernel_dtype=kernel_dtype,
        acc_type=acc_type,
        identity_for_transpose_k_hbm=identity_for_transpose_k_hbm,
        identity_for_transpose_p_hbm=identity_for_transpose_p_hbm,
        batch_id=batch_id,
        head_id=head_id,
        q_h_per_k_h=q_h_per_k_h,
        softmax_scale=softmax_scale,
        n_small_in_large_q_tile=n_small_in_large_q_tile,
        INNER_Q_TILE_SIZE=INNER_Q_TILE_SIZE,
        LARGE_KV_TILE_SIZE=LARGE_KV_TILE_SIZE,
        B_F_SIZE=B_F_SIZE,
        B_D_SIZE=B_D_SIZE,
    )

    prefill_active_tokens_and_epilogue(
        o=o,
        query=query,
        key=key,
        value=value,
        active_mask=active_mask,
        softmax_scale=softmax_scale,
        m_buffer=m_buffer,
        l_buffer=l_buffer,
        o_buffer=o_buffer,
        ACTIVE_Q_TILE_SIZE=INNER_Q_TILE_SIZE,
        seqlen_q=seqlen_q,
        batch_id=batch_id,
        head_id=head_id,
        q_h_per_k_h=q_h_per_k_h,
        kernel_dtype=kernel_dtype,
        acc_type=acc_type,
        B_F_SIZE=B_F_SIZE,
        B_D_SIZE=B_D_SIZE,
    )
    return o


@nki.compiler.skip_middle_end_transformations
@nki.jit(experimental_flags="experimental-native-scalar-support")
def flash_paged_attn_blockspase_prefill_dynamic(
    query,
    key,
    value,
    key_cache,
    value_cache,
    tile_q_indices,
    tile_block_tables,
    tile_masks,
    active_mask,
    num_dynamic_loop_steps,
    dynamic_loop_unroll_factor=1,
    softmax_scale=None,
    mixed_precision=True,
):
    b, h, k_h, seqlen_q, d, LARGE_KV_TILE_SIZE, INNER_Q_TILE_SIZE = check_input_shapes(
        query, key, value, key_cache, value_cache, tile_masks, active_mask, False
    )
    batch_id = nl.program_id(axis=0)  # equals 0
    head_id = nl.program_id(axis=1)
    kernel_dtype = nl.bfloat16 if mixed_precision else query.dtype
    acc_type = np.dtype(np.float32) if mixed_precision else kernel_dtype
    o = nl.ndarray((b, h, seqlen_q, d), dtype=query.dtype, buffer=nl.shared_hbm)
    softmax_scale = softmax_scale or (1.0 / (d**0.5))
    q_h_per_k_h = h // k_h
    B_D_SIZE = d

    key_cache, value_cache, block_size_tiling_factor, block_size = (
        prepare_kv_block_dim_tiling(key_cache, value_cache, LARGE_KV_TILE_SIZE)
    )
    num_blocks_per_large_tile = LARGE_KV_TILE_SIZE // block_size

    assert num_dynamic_loop_steps.dtype == nl.int32
    B_F_SIZE = B_FMAX_SIZE
    _, LARGE_Q_TILE_SIZE, LARGE_KV_TILE_SIZE = tile_masks.shape
    assert (
        LARGE_KV_TILE_SIZE % B_F_SIZE == 0
    ), f"Need LARGE_KV_TILE_SIZE ({LARGE_KV_TILE_SIZE=}) to be divisible by ({B_F_SIZE=})"
    n_small_in_large_q_tile = LARGE_Q_TILE_SIZE // INNER_Q_TILE_SIZE
    NEG_INF = -9984.0  # Magic number to replace -inf similar to what Tensorizer uses
    # =============== Global Flash Attention accumulators ====================== #
    o_buffer = nl.ndarray(
        (seqlen_q, q_h_per_k_h, B_D_SIZE),
        dtype=acc_type,
        buffer=nl.hbm,
    )
    m_buffer = nl.ndarray(
        (seqlen_q, q_h_per_k_h, 1),
        dtype=acc_type,
        buffer=nl.hbm,
    )
    # FIXME: rename l_buffer as sumexp_buffer
    l_buffer = nl.ndarray(
        (seqlen_q, q_h_per_k_h, 1),
        dtype=acc_type,
        buffer=nl.hbm,
    )

    for i in nl.affine_range(ceil_div(seqlen_q, INNER_Q_TILE_SIZE)):
        i_x, i_y, i_z = nl.mgrid[:INNER_Q_TILE_SIZE, :q_h_per_k_h, :B_D_SIZE]
        i_x = i_x + i * INNER_Q_TILE_SIZE
        nl.store(dst=o_buffer[i_x, i_y, i_z], value=0.0, mask=(i_x < seqlen_q))
        i_x, i_y, i_z = nl.mgrid[:INNER_Q_TILE_SIZE, :q_h_per_k_h, :1]
        i_x = i_x + i * INNER_Q_TILE_SIZE
        nl.store(dst=m_buffer[i_x, i_y, i_z], value=NEG_INF, mask=(i_x < seqlen_q))
        nl.store(dst=l_buffer[i_x, i_y, i_z], value=0, mask=(i_x < seqlen_q))

    # =============== Global Flash Attention accumulators END ================== #

    # transpose identity matrix on hbm
    identity_for_transpose_k_hbm = nl.shared_constant(
        np.identity(n=B_P_SIZE, dtype=np.uint8),
        dtype=kernel_dtype,
    )
    identity_for_transpose_p_hbm = nl.shared_constant(
        np.identity(n=INNER_Q_TILE_SIZE, dtype=np.uint8),
        dtype=kernel_dtype,
    )

    cond_dtype = np.int32
    cond = nl.ndarray((1, 1), buffer=nl.hbm, dtype=cond_dtype)
    loop_index = nl.zeros((1, 1), dtype=np.int32)
    cond_var = True
    num_dynamic_loop_steps_sbuf = nl.load(num_dynamic_loop_steps)
    assert num_dynamic_loop_steps_sbuf.shape == (1, 1)

    while scalar(cond_var):
        MAX_NUM_TILE, LARGE_Q_TILE_SIZE, LARGE_KV_TILE_SIZE = tile_masks.shape
        tile_q_indices_sbuf = prepare_q_indices_range(
            tile_q_indices,
            INNER_Q_TILE_SIZE,
            loop_index=loop_index,
            num_tiles_per_step=dynamic_loop_unroll_factor,
        )
        block_tables_sbuf = load_indices_for_loop_step(
            tile_block_tables,
            loop_index,
            dynamic_loop_unroll_factor,
        )
        block_tables_sbuf = transform_block_tables_for_indirect_load(
            block_tables_sbuf,
            block_size_tiling_factor=block_size_tiling_factor,
            num_head=k_h,
            head_id=head_id,
        )
        tile_masks = tile_masks.reshape(
            (MAX_NUM_TILE * LARGE_Q_TILE_SIZE, LARGE_KV_TILE_SIZE)
        )
        cur_masks = load_indices_for_loop_step(
            tile_masks,
            loop_index,
            LARGE_Q_TILE_SIZE * dynamic_loop_unroll_factor,
            partition_size=INNER_Q_TILE_SIZE,
        )
        prefill_prior_tokens(
            num_tiles_unrolled=dynamic_loop_unroll_factor,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            m_buffer=m_buffer,
            l_buffer=l_buffer,
            o_buffer=o_buffer,
            tile_q_indices_sbuf=tile_q_indices_sbuf,
            cur_masks=cur_masks,
            tile_masks=None,
            block_tables_sbuf=block_tables_sbuf,
            num_blocks_per_large_tile=num_blocks_per_large_tile,
            block_size=block_size,
            kernel_dtype=kernel_dtype,
            acc_type=acc_type,
            identity_for_transpose_k_hbm=identity_for_transpose_k_hbm,
            identity_for_transpose_p_hbm=identity_for_transpose_p_hbm,
            batch_id=batch_id,
            head_id=head_id,
            q_h_per_k_h=q_h_per_k_h,
            softmax_scale=softmax_scale,
            n_small_in_large_q_tile=n_small_in_large_q_tile,
            INNER_Q_TILE_SIZE=INNER_Q_TILE_SIZE,
            LARGE_KV_TILE_SIZE=LARGE_KV_TILE_SIZE,
            B_F_SIZE=B_F_SIZE,
            B_D_SIZE=B_D_SIZE,
        )
        # update loop_index
        loop_index[...] = nl.add(loop_index[...], 1)
        # update conditions
        cond_next = nisa.tensor_tensor(
            loop_index,
            num_dynamic_loop_steps_sbuf,
            nl.less,
            dtype=nl.int32,
        )
        nl.store(dst=cond, value=cond_next)
        cond_var = cond

    prefill_active_tokens_and_epilogue(
        o=o,
        query=query,
        key=key,
        value=value,
        active_mask=active_mask,
        softmax_scale=softmax_scale,
        m_buffer=m_buffer,
        l_buffer=l_buffer,
        o_buffer=o_buffer,
        ACTIVE_Q_TILE_SIZE=INNER_Q_TILE_SIZE,
        seqlen_q=seqlen_q,
        batch_id=batch_id,
        head_id=head_id,
        q_h_per_k_h=q_h_per_k_h,
        kernel_dtype=kernel_dtype,
        acc_type=acc_type,
        B_F_SIZE=B_F_SIZE,
        B_D_SIZE=B_D_SIZE,
    )
    return o


@nki.compiler.skip_middle_end_transformations
@nki.jit
def flash_paged_attn_blockspase_decode_static(
    query,
    key,
    value,
    key_cache,
    value_cache,
    tile_q_indices,
    tile_block_tables,
    tile_masks,
    active_mask,
    last_tile_indices,
    q_update_pred=None,
    softmax_scale=None,
    mixed_precision=True,
):
    b, h, k_h, seqlen_q, d, LARGE_KV_TILE_SIZE, _ = check_input_shapes(
        query, key, value, key_cache, value_cache, tile_masks, active_mask, True
    )
    batch_id = nl.program_id(axis=0)  # equals 0
    head_id = nl.program_id(axis=1)
    kernel_dtype = nl.bfloat16 if mixed_precision else query.dtype
    acc_type = np.dtype(np.float32) if mixed_precision else kernel_dtype
    o = nl.ndarray((b, h, seqlen_q, d), dtype=query.dtype, buffer=nl.shared_hbm)
    softmax_scale = softmax_scale or (1.0 / (d**0.5))
    q_h_per_k_h = h // k_h
    B_D_SIZE = d

    key_cache, value_cache, block_size_tiling_factor, block_size = (
        prepare_kv_block_dim_tiling(key_cache, value_cache, LARGE_KV_TILE_SIZE)
    )
    num_blocks_per_large_tile = LARGE_KV_TILE_SIZE // block_size

    MAX_NUM_TILE = tile_masks.shape[1]
    # =============== Global Flash Attention accumulators ====================== #
    lmo_buffer = nl.ndarray(
        (par_dim(MAX_NUM_TILE), q_h_per_k_h, B_D_SIZE + 2),
        dtype=acc_type,
        buffer=nl.hbm,
    )
    # =============== Global Flash Attention accumulators END ================== #

    # transpose identity matrix for m_buffer on hbm
    identity_for_transpose_m_step1_hbm = nl.shared_constant(
        np.identity(n=B_P_SIZE, dtype=np.uint8),
        dtype=acc_type,
    )
    identity_for_transpose_m_step2_hbm = nl.shared_constant(
        np.identity(n=q_h_per_k_h, dtype=np.uint8),
        dtype=acc_type,
    )
    identity_for_transpose_k_hbm = nl.shared_constant(
        np.identity(n=B_P_SIZE, dtype=np.uint8),
        dtype=kernel_dtype,
    )

    last_tile_indices_sbuf = nl.load(last_tile_indices)
    if q_update_pred is None:
        q_update_pred_hbm = prepare_q_update_pred(last_tile_indices_sbuf, MAX_NUM_TILE)
        q_update_pred = q_update_pred_hbm.reshape((1, MAX_NUM_TILE))

    MULTI_BUFFER_SIZE = 2
    MULTI_BUFFER_SIZE = min(MULTI_BUFFER_SIZE, MAX_NUM_TILE)
    _, MAX_NUM_TILE, num_k_tiles = tile_masks.shape
    assert tile_masks.shape[0] == B_P_SIZE
    assert q_h_per_k_h <= B_P_SIZE

    num_loads = num_blocks_per_large_tile // B_P_SIZE
    assert num_k_tiles == num_loads * block_size

    block_tables_sbuf = load_indices(
        tile_block_tables,
    )
    block_tables_sbuf = transform_block_tables_for_indirect_load(
        block_tables_sbuf,
        block_size_tiling_factor=block_size_tiling_factor,
        num_head=k_h,
        head_id=head_id,
    )

    # load all q and multiply scale
    query_sbuf = load_decode_query(
        query=query,
        tile_q_indices=tile_q_indices,
        q_indices_region=None,
        softmax_scale=softmax_scale,
        batch_id=batch_id,
        head_id=head_id,
        q_h_per_k_h=q_h_per_k_h,
        B_D_SIZE=B_D_SIZE,
        kernel_dtype=kernel_dtype,
    )
    # prepare tile_mask
    tile_mask_sbuf = nl.ndarray(
        (par_dim(B_P_SIZE), MAX_NUM_TILE, num_k_tiles),
        dtype=tile_masks.dtype,
    )
    tile_mask_sbuf[...] = nl.load(tile_masks)

    q_update_pred_sbuf, q_update_pred_broadcast = load_and_broadcast_q_update_preds(
        q_update_pred,
        B_D_SIZE,
        loop_index=None,
    )
    decode_prior_tokens(
        num_tiles_unrolled=MAX_NUM_TILE,
        query_sbuf=query_sbuf,
        key_cache=key_cache,
        value_cache=value_cache,
        lmo_buffer=lmo_buffer,
        m_next_tile=None,
        l_next_tile=None,
        o_next_tile=None,
        q_offsets=None,
        block_tables_sbuf=block_tables_sbuf,
        tile_mask_sbuf=tile_mask_sbuf,
        q_update_pred_sbuf=q_update_pred_sbuf,
        q_update_pred_broadcast=q_update_pred_broadcast,
        num_blocks_per_large_tile=num_blocks_per_large_tile,
        block_size=block_size,
        kernel_dtype=kernel_dtype,
        acc_type=acc_type,
        q_h_per_k_h=q_h_per_k_h,
        B_D_SIZE=B_D_SIZE,
        MULTI_BUFFER_SIZE=MULTI_BUFFER_SIZE,
        identity_for_transpose_m_step1_hbm=identity_for_transpose_m_step1_hbm,
        identity_for_transpose_m_step2_hbm=identity_for_transpose_m_step2_hbm,
        identity_for_transpose_k_hbm=identity_for_transpose_k_hbm,
    )

    decode_active_tokens_and_epilogue(
        o=o,
        query=query,
        key=key,
        value=value,
        lmo_buffer=lmo_buffer,
        softmax_scale=softmax_scale,
        last_tile_indices_sbuf=last_tile_indices_sbuf,
        seqlen_q=seqlen_q,
        batch_id=batch_id,
        head_id=head_id,
        q_h_per_k_h=q_h_per_k_h,
        kernel_dtype=kernel_dtype,
        acc_type=acc_type,
        B_D_SIZE=B_D_SIZE,
    )
    return o


@nki.compiler.skip_middle_end_transformations
@nki.jit(experimental_flags="experimental-native-scalar-support")
def flash_paged_attn_blockspase_decode_dynamic(
    query,
    key,
    value,
    key_cache,
    value_cache,
    tile_q_indices,
    tile_block_tables,
    tile_masks,
    active_mask,
    num_dynamic_loop_steps,
    last_tile_indices,
    q_update_pred=None,
    dynamic_loop_unroll_factor=1,
    softmax_scale=None,
    mixed_precision=True,
):
    b, h, k_h, seqlen_q, d, LARGE_KV_TILE_SIZE, _ = check_input_shapes(
        query, key, value, key_cache, value_cache, tile_masks, active_mask, True
    )
    batch_id = nl.program_id(axis=0)  # equals 0
    head_id = nl.program_id(axis=1)
    kernel_dtype = nl.bfloat16 if mixed_precision else query.dtype
    acc_type = np.dtype(np.float32) if mixed_precision else kernel_dtype
    o = nl.ndarray((b, h, seqlen_q, d), dtype=query.dtype, buffer=nl.shared_hbm)
    softmax_scale = softmax_scale or (1.0 / (d**0.5))
    q_h_per_k_h = h // k_h
    B_D_SIZE = d

    key_cache, value_cache, block_size_tiling_factor, block_size = (
        prepare_kv_block_dim_tiling(key_cache, value_cache, LARGE_KV_TILE_SIZE)
    )
    num_blocks_per_large_tile = LARGE_KV_TILE_SIZE // block_size

    MAX_NUM_TILE = tile_masks.shape[1]
    assert MAX_NUM_TILE % dynamic_loop_unroll_factor == 0
    # =============== Global Flash Attention accumulators ====================== #
    lmo_buffer = nl.ndarray(
        (par_dim(MAX_NUM_TILE), q_h_per_k_h, B_D_SIZE + 2),
        dtype=acc_type,
        buffer=nl.hbm,
    )
    # =============== Global Flash Attention accumulators END ================== #
    o_next_tile = nl.ndarray(
        (par_dim(B_D_SIZE), q_h_per_k_h),
        dtype=acc_type,
        buffer=nl.hbm,
    )
    m_next_tile = nl.ndarray(
        (par_dim(1), q_h_per_k_h),
        dtype=acc_type,
        buffer=nl.hbm,
    )
    l_next_tile = nl.ndarray(
        (par_dim(1), q_h_per_k_h),
        dtype=acc_type,
        buffer=nl.hbm,
    )
    NEG_INF = -9984.0  # Magic number to replace -inf similar to what Tensorizer uses
    nl.store(o_next_tile, 0)
    nl.store(m_next_tile, NEG_INF)
    nl.store(l_next_tile, 0)

    # transpose identity matrix for m_buffer on hbm
    identity_for_transpose_m_step1_hbm = nl.shared_constant(
        np.identity(n=B_P_SIZE, dtype=np.uint8),
        dtype=acc_type,
    )
    identity_for_transpose_m_step2_hbm = nl.shared_constant(
        np.identity(n=q_h_per_k_h, dtype=np.uint8),
        dtype=acc_type,
    )
    identity_for_transpose_k_hbm = nl.shared_constant(
        np.identity(n=B_P_SIZE, dtype=np.uint8),
        dtype=kernel_dtype,
    )

    last_tile_indices_sbuf = nl.load(last_tile_indices)
    if q_update_pred is None:
        q_update_pred_hbm = prepare_q_update_pred(last_tile_indices_sbuf, MAX_NUM_TILE)
        q_update_pred = q_update_pred_hbm.reshape(
            (MAX_NUM_TILE // dynamic_loop_unroll_factor, dynamic_loop_unroll_factor)
        )

    cond_dtype = np.int32
    cond = nl.ndarray((1, 1), buffer=nl.hbm, dtype=cond_dtype)
    loop_index = nl.zeros((1, 1), dtype=np.int32)
    cond_var = True
    num_dynamic_loop_steps_sbuf = nl.load(num_dynamic_loop_steps)
    assert num_dynamic_loop_steps_sbuf.shape == (1, 1)

    while scalar(cond_var):
        MULTI_BUFFER_SIZE = 2
        MULTI_BUFFER_SIZE = min(MULTI_BUFFER_SIZE, dynamic_loop_unroll_factor)
        _, MAX_NUM_TILE, num_k_tiles = tile_masks.shape
        assert tile_masks.shape[0] == B_P_SIZE
        assert q_h_per_k_h <= B_P_SIZE

        num_loads = num_blocks_per_large_tile // B_P_SIZE
        assert num_k_tiles == num_loads * block_size

        block_tables_sbuf = load_indices_for_loop_step(
            tile_block_tables,
            loop_index,
            dynamic_loop_unroll_factor,
        )
        block_tables_sbuf = transform_block_tables_for_indirect_load(
            block_tables_sbuf,
            block_size_tiling_factor=block_size_tiling_factor,
            num_head=k_h,
            head_id=head_id,
        )

        q_offsets = prepare_decode_offsets(loop_index, dynamic_loop_unroll_factor)

        # load all q and multiply scale
        query_sbuf = load_decode_query(
            query=query,
            tile_q_indices=tile_q_indices,
            q_indices_region=q_offsets,
            softmax_scale=softmax_scale,
            batch_id=batch_id,
            head_id=head_id,
            q_h_per_k_h=q_h_per_k_h,
            B_D_SIZE=B_D_SIZE,
            kernel_dtype=kernel_dtype,
        )
        # prepare tile_mask
        tile_masks = tile_masks.reshape(
            (
                B_P_SIZE,
                MAX_NUM_TILE // dynamic_loop_unroll_factor,
                dynamic_loop_unroll_factor,
                num_k_tiles,
            )
        )
        tile_mask_sbuf = nl.ndarray(
            (par_dim(B_P_SIZE), dynamic_loop_unroll_factor, num_k_tiles),
            dtype=tile_masks.dtype,
        )
        i_p = nl.arange(B_P_SIZE)[:, None, None]
        i_q = nl.arange(dynamic_loop_unroll_factor)[None, :, None]
        i_k = nl.arange(num_k_tiles)[None, None, :]
        tile_mask_sbuf[i_p, i_q, i_k] = nl.load(
            tile_masks[i_p, loop_index[0, 0], i_q, i_k]
        )

        q_update_pred_sbuf, q_update_pred_broadcast = load_and_broadcast_q_update_preds(
            q_update_pred,
            B_D_SIZE,
            loop_index=loop_index,
        )
        decode_prior_tokens(
            num_tiles_unrolled=dynamic_loop_unroll_factor,
            query_sbuf=query_sbuf,
            key_cache=key_cache,
            value_cache=value_cache,
            lmo_buffer=lmo_buffer,
            m_next_tile=m_next_tile,
            l_next_tile=l_next_tile,
            o_next_tile=o_next_tile,
            q_offsets=q_offsets,
            block_tables_sbuf=block_tables_sbuf,
            tile_mask_sbuf=tile_mask_sbuf,
            q_update_pred_sbuf=q_update_pred_sbuf,
            q_update_pred_broadcast=q_update_pred_broadcast,
            num_blocks_per_large_tile=num_blocks_per_large_tile,
            block_size=block_size,
            kernel_dtype=kernel_dtype,
            acc_type=acc_type,
            q_h_per_k_h=q_h_per_k_h,
            B_D_SIZE=B_D_SIZE,
            MULTI_BUFFER_SIZE=MULTI_BUFFER_SIZE,
            identity_for_transpose_m_step1_hbm=identity_for_transpose_m_step1_hbm,
            identity_for_transpose_m_step2_hbm=identity_for_transpose_m_step2_hbm,
            identity_for_transpose_k_hbm=identity_for_transpose_k_hbm,
        )
        # update loop_index
        loop_index[...] = nl.add(loop_index[...], 1)
        # update conditions
        cond_next = nisa.tensor_tensor(
            loop_index,
            num_dynamic_loop_steps_sbuf,
            nl.less,
            dtype=nl.int32,
        )
        nl.store(dst=cond, value=cond_next)
        cond_var = cond

    decode_active_tokens_and_epilogue(
        o=o,
        query=query,
        key=key,
        value=value,
        lmo_buffer=lmo_buffer,
        softmax_scale=softmax_scale,
        last_tile_indices_sbuf=last_tile_indices_sbuf,
        seqlen_q=seqlen_q,
        batch_id=batch_id,
        head_id=head_id,
        q_h_per_k_h=q_h_per_k_h,
        kernel_dtype=kernel_dtype,
        acc_type=acc_type,
        B_D_SIZE=B_D_SIZE,
    )
    return o


def flash_attn_varlen_blocksparse_nkifunc(
    query,
    key,
    value,
    key_cache,
    value_cache,
    tile_q_indices,
    tile_block_tables,
    tile_masks,
    active_mask,
    num_dynamic_loop_steps,
    last_tile_indices,
    q_update_pred=None,
    dynamic_loop_unroll_factor=1,
    n_kv_head=None,
    head_size=None,
    mixed_precision=True,
    decode_mode=False,
):
    """
    Flash pagedAttention kernel for batched variable-length sequences

    IO tensor layouts:
      - query: shape (1, n_heads, seq_q, d)
      - key:   shape (1, n_kv_heads, d, seq_k)
      - value: shape (1, n_kv_heads, seq_v, d)
      - key_cache: (max_num_blocks, n_kv_heads, block_size, d)
      - value_cache: (max_num_blocks, n_kv_heads, block_size, d)
      - tile_q_indices: (num_large_tiles, large_tile_size_q)
      - tile_block_tables: (num_large_tiles, num_block_per_large_tile)
      - tile_masks: (num_large_tiles, large_tile_size_q, large_tile_size_k) if not decode_mode
          else (num_large_tiles, large_tile_size_q, large_tile_size_k)
      - active_mask: (seq_q, seq_q)

      - This kernel requires seq_k == seq_v
      - We use continuous batching by default, so the batch dimension is always 1, and different
        requests are concatenated along sequence dimension.
      - We use paged cache blocks (key_cache, value_cache) to store KV cache.

    IO tensor dtypes:
      - This kernel assumes all IO tensors have the same dtype except for block_tables (uint32) and mask (uint8)
      - If mixed_percision is True, then all Tensor Engine operation will be performed in
        bfloat16 and accumulation will be performed in float32. Otherwise the intermediates
        will be in the same type as the inputs.

    Compile-time Constants:
      - sequence_parallel_group: sequence parallel group to shard the cache blocks, List[int].
      - softmax_scale: scaling for softmax, is None, default is `1.0/(d**0.5)`
      - mixed_precision: flag to set non-matmul ops in fp32 precision, defualt is set to `true`,
          if false, we use same precision as input types

    GQA support Notes:
      the spmd kernel for launching kernel should be on kv_heads instead of nheads

    Example usage:
      MHA: q: [b, h, d, s], k: [b, h, d, s], v: [b, h, s, d]
        usage: `flash_fwd[b, h](q, k, v, ...)`
      GQA: q: [b, h, d, s], k: [b, kv_h, d, s], v: [b, kv_h, s, d]
        usage: `flash_fwd[b, kv_h](q, k, v, ...)`
    """
    if n_kv_head is None:
        n_kv_head = key_cache.shape[1]
    assert key_cache.shape[1] == n_kv_head
    if head_size is None:
        head_size = key_cache.shape[-1]
    if dynamic_loop_unroll_factor:
        assert num_dynamic_loop_steps is not None
        assert n_kv_head == 1, f"dynamic loop does not support SPMD launch"
        if decode_mode:
            kwargs = dict(
                query=query,
                key=key,
                value=value,
                key_cache=key_cache,
                value_cache=value_cache,
                tile_q_indices=tile_q_indices,
                tile_block_tables=tile_block_tables,
                tile_masks=tile_masks,
                active_mask=active_mask,
                q_update_pred=q_update_pred,
                last_tile_indices=last_tile_indices,
                num_dynamic_loop_steps=num_dynamic_loop_steps,
                dynamic_loop_unroll_factor=dynamic_loop_unroll_factor,
                softmax_scale=1.0 / (head_size**0.5),
                mixed_precision=mixed_precision,
            )
            return flash_paged_attn_blockspase_decode_dynamic[1, n_kv_head](**kwargs)
        else:
            kwargs = dict(
                query=query,
                key=key,
                value=value,
                key_cache=key_cache,
                value_cache=value_cache,
                tile_q_indices=tile_q_indices,
                tile_block_tables=tile_block_tables,
                tile_masks=tile_masks,
                active_mask=active_mask,
                num_dynamic_loop_steps=num_dynamic_loop_steps,
                dynamic_loop_unroll_factor=dynamic_loop_unroll_factor,
                softmax_scale=1.0 / (head_size**0.5),
                mixed_precision=mixed_precision,
            )
            return flash_paged_attn_blockspase_prefill_dynamic[1, n_kv_head](**kwargs)
    else:
        assert num_dynamic_loop_steps is None
        if decode_mode:
            kwargs = dict(
                query=query,
                key=key,
                value=value,
                key_cache=key_cache,
                value_cache=value_cache,
                tile_q_indices=tile_q_indices,
                tile_block_tables=tile_block_tables,
                tile_masks=tile_masks,
                active_mask=active_mask,
                q_update_pred=q_update_pred,
                last_tile_indices=last_tile_indices,
                softmax_scale=1.0 / (head_size**0.5),
                mixed_precision=mixed_precision,
            )
            return flash_paged_attn_blockspase_decode_static[1, n_kv_head](**kwargs)
        else:
            kwargs = dict(
                query=query,
                key=key,
                value=value,
                key_cache=key_cache,
                value_cache=value_cache,
                tile_q_indices=tile_q_indices,
                tile_block_tables=tile_block_tables,
                tile_masks=tile_masks,
                active_mask=active_mask,
                softmax_scale=1.0 / (head_size**0.5),
                mixed_precision=mixed_precision,
            )
            return flash_paged_attn_blockspase_prefill_static[1, n_kv_head](**kwargs)
