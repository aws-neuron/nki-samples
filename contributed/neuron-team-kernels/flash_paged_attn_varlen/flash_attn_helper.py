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

from paged_cache import load_kv_tile_from_cache
from utils import (
    B_P_SIZE,
    ceil_div,
    pad_to_multiple,
    broadcast_partition_with_PE,
    PF_transpose_with_PE,
    PF_transpose_with_PE,
    create_identity_for_transpose,
    load_indices,
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


def transpose_p_local(
    p_local_transposed,
    p_local,
    identity_for_transpose,
    Q_TILE_SIZE,
    LARGE_KV_TILE_SIZE,
    B_F_SIZE=512,
):
    B_P_SIZE = 128
    REDUCTION_SIZE = min(B_P_SIZE, LARGE_KV_TILE_SIZE)
    for i in nl.affine_range(LARGE_KV_TILE_SIZE // B_F_SIZE):
        if nisa.get_nc_version() == nisa.nc_version.gen3:
            p_local_t_tmp = nl.ndarray(
                (par_dim(REDUCTION_SIZE), B_F_SIZE // REDUCTION_SIZE * Q_TILE_SIZE),
                buffer=nl.sbuf,
                dtype=p_local.dtype,
            )
        else:
            p_local_t_tmp = nl.ndarray(
                (par_dim(REDUCTION_SIZE), B_F_SIZE // REDUCTION_SIZE * Q_TILE_SIZE),
                buffer=nl.psum,
                dtype=np.float32,
            )

        for j in nl.affine_range(B_F_SIZE // REDUCTION_SIZE):
            j_128_slice = nl.ds(j * Q_TILE_SIZE, Q_TILE_SIZE)
            i_j_128_slice = nl.ds(i * B_F_SIZE + j * REDUCTION_SIZE, REDUCTION_SIZE)

            if nisa.get_nc_version() == nisa.nc_version.gen3:
                p_local_t_tmp[:, j_128_slice] = nisa.dma_transpose(
                    p_local[:, i_j_128_slice]
                )
            else:
                p_local_t_tmp[:, j_128_slice] = nisa.nc_matmul(
                    p_local[:, i_j_128_slice],
                    identity_for_transpose,
                    is_moving_onezero=True,
                    is_transpose=True,
                )

        p_local_transposed[
            :,
            nl.ds(
                i * (B_F_SIZE // REDUCTION_SIZE * Q_TILE_SIZE),
                (B_F_SIZE // REDUCTION_SIZE * Q_TILE_SIZE),
            ),
        ] = nl.copy(p_local_t_tmp, dtype=p_local_transposed.dtype)


def _flash_attention_core(
    q_local_tile,
    k,
    v,
    o_buffer,
    l_buffer,
    m_buffer,
    kernel_dtype,
    acc_type,
    tile_mask,
    use_causal_mask,
    identity_for_transpose,
    q_tile_idx=None,
    Q_TILE_SIZE=128,
    LARGE_KV_TILE_SIZE=2048,
    B_F_SIZE=512,
    B_D_SIZE=128,
):
    """
    The flash attention core function to calculate self attention between a tile
    of q and a block of K and V.
    q_local_tile: (B_D_SIZE, Q_TILE_SIZE)
    k: (B_D_SIZE, LARGE_KV_TILE_SIZE)
    v: (B_P_SIZE, LARGE_KV_TILE_SIZE // B_P_SIZE, B_D_SIZE)
    The results are stored in the following three buffers
    o_buffer: (Q_TILE_SIZE, B_D_SIZE)
    l_buffer: (Q_TILE_SIZE, 1)
    m_buffer: (Q_TILE_SIZE, 1)
    """
    B_P_SIZE = 128
    assert (
        LARGE_KV_TILE_SIZE % B_P_SIZE == 0
    ), f"{LARGE_KV_TILE_SIZE=} not divisive by {B_P_SIZE=}"
    assert (
        LARGE_KV_TILE_SIZE % B_F_SIZE == 0
    ), f"{LARGE_KV_TILE_SIZE=} not divisive by {B_F_SIZE=}"
    num_k_tile_per_large_tile = LARGE_KV_TILE_SIZE // B_F_SIZE

    # mask are used to only apply computation to the lower half of the matrix,
    # which reduce the arithmetic intensity by half
    qk_res_buf = nl.ndarray(
        (par_dim(Q_TILE_SIZE), LARGE_KV_TILE_SIZE),
        buffer=nl.sbuf,
        dtype=acc_type,
    )
    max_local = nl.ndarray(
        (par_dim(Q_TILE_SIZE), num_k_tile_per_large_tile),
        dtype=acc_type,
    )
    for k_i in nl.affine_range(num_k_tile_per_large_tile):
        k_i_b_f_slice = nl.ds(k_i * B_F_SIZE, B_F_SIZE)

        if use_causal_mask:
            multiplication_required_selection = (
                q_tile_idx * Q_TILE_SIZE >= k_i * B_F_SIZE
            )
        else:
            multiplication_required_selection = True

        if multiplication_required_selection:
            qk_psum = nl.ndarray(
                (par_dim(Q_TILE_SIZE), B_F_SIZE), dtype=np.float32, buffer=nl.psum
            )  # (128, 512)
            qk_psum[:, :] = nl.matmul(
                q_local_tile, k[:, k_i_b_f_slice], transpose_x=True
            )  # (p(128), 512)
            qk_res_buf[:, k_i_b_f_slice] = nl.where(
                tile_mask[:, k_i_b_f_slice],
                qk_psum[:, nl.ds(0, B_F_SIZE)],
                -9984.0,
                dtype=acc_type,
            )
        else:
            qk_res_buf[:, k_i_b_f_slice] = -9984.0

        # Calculate max of the current tile
        max_local[:, k_i] = nisa.tensor_reduce(
            np.max,
            qk_res_buf[:, k_i_b_f_slice],
            axis=(1,),
            dtype=acc_type,
            negate=False,
        )

    max_ = nisa.tensor_reduce(
        np.max,
        max_local[:, :],
        axis=(1,),
        dtype=acc_type,
        negate=False,
    )

    o_previous_scaled = nl.ndarray(
        (par_dim(Q_TILE_SIZE), B_D_SIZE),
        dtype=o_buffer.dtype,
    )

    m_previous = nl.copy(m_buffer[:, 0])
    m_buffer[:, 0] = nl.maximum(m_previous, max_)  # (128,1)

    m_current = m_buffer[:, 0]
    # Compute scaling factor
    alpha = nisa.activation(
        np.exp,
        m_previous,
        bias=-1 * m_current,
        scale=1.0,
    )
    o_previous_scaled[...] = nl.multiply(o_buffer[:, :], alpha)

    p_local = nl.ndarray(
        (par_dim(Q_TILE_SIZE), LARGE_KV_TILE_SIZE),
        dtype=kernel_dtype,
    )
    REDUCTION_TILE = min(2048, LARGE_KV_TILE_SIZE // 2)

    p_partial_sum = nl.ndarray(
        (par_dim(Q_TILE_SIZE), LARGE_KV_TILE_SIZE // REDUCTION_TILE),
        dtype=acc_type,
    )

    for k_r_i in nl.affine_range(LARGE_KV_TILE_SIZE // REDUCTION_TILE):
        k_r_i_reduce_slice = nl.ds(k_r_i * REDUCTION_TILE, REDUCTION_TILE)

        # compute exp(qk - max)
        # Compute partial row - tile sum of exp(qk - max))
        # FIXME : Use activation accumulate to accumulate over k_r_i loop ?
        p_local[:, k_r_i_reduce_slice] = nisa.activation_reduce(
            np.exp,
            qk_res_buf[:, k_r_i_reduce_slice],
            bias=-1 * m_current,
            scale=1.0,
            reduce_op=nl.add,
            reduce_res=p_partial_sum[:, k_r_i],
            dtype=kernel_dtype,
        )

    ps = nl.sum(p_partial_sum, axis=1, dtype=acc_type)

    p_local_transposed = nl.ndarray(
        (par_dim(B_P_SIZE), LARGE_KV_TILE_SIZE // B_P_SIZE * Q_TILE_SIZE),
        dtype=kernel_dtype,
    )
    transpose_p_local(
        p_local_transposed=p_local_transposed,
        p_local=p_local,
        identity_for_transpose=identity_for_transpose,
        Q_TILE_SIZE=Q_TILE_SIZE,
        LARGE_KV_TILE_SIZE=LARGE_KV_TILE_SIZE,
        B_F_SIZE=B_F_SIZE,
    )

    pv_psum = nl.zeros(
        (par_dim(Q_TILE_SIZE), B_D_SIZE),
        dtype=np.float32,
        buffer=nl.psum,
    )
    for k_i in nl.affine_range(LARGE_KV_TILE_SIZE // B_P_SIZE):
        pv_psum[:, :] += nl.matmul(
            p_local_transposed[:, nl.ds(k_i * Q_TILE_SIZE, Q_TILE_SIZE)],
            v[:, nl.ds(k_i * B_D_SIZE, B_D_SIZE)],
            transpose_x=True,
        )  # (128, 128) (p(Br), d)

    o_buffer[:, :] = nl.add(o_previous_scaled, pv_psum)

    l_prev = l_buffer[:, 0] * alpha
    l_buffer[:, 0] = l_prev + ps


def _flash_attention_core_kq_matmul(
    q_local_tile,
    o_buffer_sbuf,
    l_buffer_sbuf,
    m_buffer_sbuf,
    kernel_dtype,
    acc_type,
    tile_mask,
    q_update_pred_sbuf,
    q_update_pred_broadcast,
    key_cache,
    value_cache,
    block_tables_sbuf,
    large_tile_idx,
    MULTI_BUFFER_SIZE,
    num_blocks_per_large_tile,
    block_size,
    k_load_buffer,
    v_load_buffer,
    kq_res_ones,
    sumexp_ones,
    identity_for_transpose_k,
    identity_for_transpose_m_step1,
    identity_for_transpose_m_step2,
):
    """
    The flash attention core function to calculate self attention between a tile
    of q and a block of K and V.
    Input:
    q_local_tile: (B_D_SIZE, q_h_per_k_h)
    k: (B_D_SIZE, kv_tile_size)
    v: (B_P_SIZE, kv_tile_size // B_P_SIZE, B_D_SIZE)
    The results are stored in the following three buffers
    o_buffer_sbuf: (B_D_SIZE, NUM_LARGE_TILE, q_h_per_k_h)
    l_buffer_sbuf: (1, NUM_LAEGE_TILE, q_h_per_k_h)
    m_buffer_sbuf: (1, NUM_LAEGE_TILE, q_h_per_k_h)
    """
    B_P_SIZE = 128
    B_D_SIZE, NUM_LARGE_TILE, Q_TILE_SIZE = q_local_tile.shape
    assert num_blocks_per_large_tile % B_P_SIZE == 0
    num_loads = num_blocks_per_large_tile // B_P_SIZE
    assert Q_TILE_SIZE <= B_P_SIZE, f"{Q_TILE_SIZE=} > {B_P_SIZE=}"
    k = k_load_buffer[:, large_tile_idx % MULTI_BUFFER_SIZE]
    v = v_load_buffer[:, large_tile_idx % MULTI_BUFFER_SIZE]

    KV_TILE_SIZE = k.shape[1]
    assert KV_TILE_SIZE % B_P_SIZE == 0, f"{KV_TILE_SIZE=} not divisive by {B_P_SIZE=}"
    num_k_tile = KV_TILE_SIZE // B_P_SIZE

    kq_res_buf = nl.ndarray(
        (par_dim(B_P_SIZE), num_k_tile, Q_TILE_SIZE),
        dtype=acc_type,
    )
    kq_res_buf[...] = kq_res_ones * (-9984.0)
    kq_res_psum = nl.ndarray(
        (par_dim(B_P_SIZE), num_k_tile, Q_TILE_SIZE),
        dtype=nl.float32,
        buffer=nl.psum,
    )
    for k_i in nl.affine_range(num_k_tile):
        kq_res_psum[:, k_i, :] = nl.matmul(
            k[:, nl.ds(k_i * B_P_SIZE, B_P_SIZE)],
            q_local_tile[:, large_tile_idx],
            transpose_x=True,
        )  # (p(128), Q_TILE_SIZE)
    if large_tile_idx + 1 < NUM_LARGE_TILE:
        k_load_buffer_reshaped = k_load_buffer.reshape(
            (B_D_SIZE, MULTI_BUFFER_SIZE, num_loads, block_size, B_P_SIZE),
        )
        for load_idx in nl.affine_range(num_loads):
            i_p = nl.arange(B_P_SIZE)[:, None]
            i_f = nl.arange(block_size * B_D_SIZE)[None, :]
            loaded = nl.ndarray(
                (B_P_SIZE, block_size * B_D_SIZE), dtype=key_cache.dtype
            )
            loaded[i_p, i_f] = nl.load(
                key_cache[block_tables_sbuf[i_p, load_idx, large_tile_idx + 1], i_f]
            )
            for tb_i in nl.affine_range(block_size):
                if loaded.dtype != kernel_dtype:
                    k_src = nl.copy(
                        loaded[:, nl.ds(tb_i * B_D_SIZE, B_D_SIZE)],
                        dtype=kernel_dtype,
                    )
                else:
                    k_src = loaded[:, nl.ds(tb_i * B_D_SIZE, B_D_SIZE)]
                PF_transpose_with_PE(
                    src=k_src,
                    out=k_load_buffer_reshaped[
                        :,
                        (large_tile_idx + 1) % MULTI_BUFFER_SIZE,
                        load_idx,
                        tb_i,
                    ],
                    identity_for_transpose=identity_for_transpose_k,
                )

    nisa.tensor_copy_predicated(
        src=kq_res_psum,
        dst=kq_res_buf,
        predicate=tile_mask[:, large_tile_idx],
    )

    max_partial = nl.ndarray((par_dim(B_P_SIZE), Q_TILE_SIZE), dtype=acc_type)
    # Calculate max of the current tile
    max_partial[:, :] = nisa.tensor_reduce(
        np.max,
        kq_res_buf,
        axis=(1,),
        dtype=acc_type,
        negate=False,
    )
    max_partial_transposed = nl.ndarray(
        (par_dim(Q_TILE_SIZE), B_P_SIZE), dtype=acc_type, buffer=nl.psum
    )
    PF_transpose_with_PE(
        src=max_partial,
        out=max_partial_transposed,
        identity_for_transpose=identity_for_transpose_m_step1,
        out_in_psum=True,
    )
    max_ = nisa.tensor_reduce(
        np.max,
        max_partial_transposed,
        axis=(1,),
        dtype=acc_type,
        negate=False,
        keepdims=True,
    )
    o_previous_scaled = nl.ndarray(
        (par_dim(B_D_SIZE), Q_TILE_SIZE),
        dtype=o_buffer_sbuf.dtype,
    )
    max_transposed = nl.ndarray(
        (par_dim(1), Q_TILE_SIZE),
        dtype=acc_type,
    )
    PF_transpose_with_PE(
        src=max_,
        out=max_transposed,
        identity_for_transpose=identity_for_transpose_m_step2,
    )

    m_previous = m_buffer_sbuf[:, large_tile_idx]
    m_current = nisa.tensor_tensor(m_previous, max_transposed, np.maximum)  # (128,1)

    # Compute scaling factor
    bias = nl.zeros((par_dim(1), 1), dtype=m_previous.dtype)
    alpha = nisa.activation(
        np.exp,
        m_previous - m_current,
        scale=1.0,
        bias=bias,
    )
    alpha_broadcasted = nl.ndarray(
        (par_dim(B_D_SIZE), Q_TILE_SIZE),
        dtype=acc_type,
        buffer=nl.psum,
    )
    broadcast_partition_with_PE(alpha, alpha_broadcasted, out_in_psum=True)
    o_previous_scaled[...] = nl.multiply(
        o_buffer_sbuf[:, large_tile_idx], alpha_broadcasted
    )

    max_broadcasted = nl.ndarray(
        (par_dim(B_P_SIZE), Q_TILE_SIZE, 1),
        dtype=acc_type,
        buffer=nl.psum,
    )
    broadcast_partition_with_PE(m_current, max_broadcasted[:, :, 0], out_in_psum=True)

    ps = nl.zeros(
        (par_dim(1), Q_TILE_SIZE),
        dtype=acc_type,
        buffer=nl.psum,
    )
    p_local = nl.ndarray(
        (par_dim(B_P_SIZE), num_k_tile, Q_TILE_SIZE),
        dtype=kernel_dtype,
    )
    bias = nl.zeros((par_dim(B_P_SIZE), 1), dtype=max_broadcasted.dtype)
    for k_r_i in nl.affine_range(num_k_tile):
        p_local[:, k_r_i, :] = nisa.activation(
            np.exp,
            kq_res_buf[:, k_r_i, :] - max_broadcasted,
            scale=1.0,
            bias=bias,
            dtype=kernel_dtype,
        )
        ps[:, :] += nisa.nc_matmul(
            sumexp_ones,
            p_local[:, k_r_i, :],
            is_stationary_onezero=True,
        )

    pv_psum = nl.zeros(
        (par_dim(B_D_SIZE), Q_TILE_SIZE),
        dtype=np.float32,
        buffer=nl.psum,
    )
    for k_i in nl.affine_range(num_k_tile):
        pv_psum[:, :] += nl.matmul(
            v[:, nl.ds(k_i * B_D_SIZE, B_D_SIZE)],
            p_local[:, k_i],
            transpose_x=True,
        )
    if large_tile_idx + 1 < NUM_LARGE_TILE:
        # load value cache
        for load_idx in nl.affine_range(num_loads):
            i_p = nl.arange(B_P_SIZE)[:, None]
            i_f = nl.arange(block_size * B_D_SIZE)[None, :]
            if kernel_dtype == value_cache.dtype:
                v_load_buffer[
                    i_p,
                    (large_tile_idx + 1) % MULTI_BUFFER_SIZE,
                    i_f + load_idx * block_size * B_D_SIZE,
                ] = nl.load(
                    value_cache[
                        block_tables_sbuf[i_p, load_idx, large_tile_idx + 1], i_f
                    ],
                )
            else:
                loaded = nl.ndarray(
                    (B_P_SIZE, block_size * B_D_SIZE), dtype=value_cache.dtype
                )
                loaded[...] = nl.load(
                    value_cache[
                        block_tables_sbuf[i_p, load_idx, large_tile_idx + 1], i_f
                    ],
                )
                v_load_buffer[
                    i_p,
                    (large_tile_idx + 1) % MULTI_BUFFER_SIZE,
                    i_f + load_idx * block_size * B_D_SIZE,
                ] = nl.copy(
                    loaded,
                    dtype=kernel_dtype,
                )

    m_buffer_sbuf[:, large_tile_idx] = m_current
    o_buffer_sbuf[:, large_tile_idx, :] = nl.add(o_previous_scaled, pv_psum)

    l_prev = l_buffer_sbuf[:, large_tile_idx] * alpha
    l_buffer_sbuf[:, large_tile_idx] = l_prev + ps
    nisa.tensor_copy_predicated(
        src=m_buffer_sbuf[:, large_tile_idx],
        dst=m_buffer_sbuf[:, large_tile_idx + 1],
        predicate=q_update_pred_sbuf[:, large_tile_idx],
    )
    nisa.tensor_copy_predicated(
        src=l_buffer_sbuf[:, large_tile_idx],
        dst=l_buffer_sbuf[:, large_tile_idx + 1],
        predicate=q_update_pred_sbuf[:, large_tile_idx],
    )
    nisa.tensor_copy_predicated(
        src=o_buffer_sbuf[:, large_tile_idx],
        dst=o_buffer_sbuf[:, large_tile_idx + 1],
        predicate=q_update_pred_broadcast[:, large_tile_idx],
    )


def _active_attention_core_batched(
    q,
    k,
    v,
    o_buffer,
    l_buffer,
    m_buffer,
    q_h_per_k_h,
    kernel_dtype,
    acc_type,
    TILE_SIZE=128,
    B_D_SIZE=128,
):
    """
    The flash attention core function to calculate self attention between a tile
    of q and a block of K and V.
    q: (par_dim(B_D_SIZE), q_h_per_k_h, TILE_SIZE)
    k: (par_dim(B_D_SIZE), TILE_SIZE)
    v: (par_dim(TILE_SIZE), B_D_SIZE)
    The results are stored in the following three buffers
    o_buffer: (par_dim(TILE_SIZE), q_h_per_k_h, B_D_SIZE)
    l_buffer: (par_dim(TILE_SIZE), q_h_per_k_h, 1)
    m_buffer: (par_dim(TILE_SIZE), q_h_per_k_h, 1)
    """
    qk_psum = nl.ndarray(
        (par_dim(TILE_SIZE), q_h_per_k_h, 1),
        buffer=nl.psum,
        dtype=np.float32,
    )
    ones = nl.ones((par_dim(B_D_SIZE), 1), dtype=acc_type)
    for i_q_h in nl.affine_range(q_h_per_k_h):
        qk_mul = nl.ndarray(
            (par_dim(B_D_SIZE), TILE_SIZE),
            dtype=acc_type,
        )
        qk_mul[...] = nl.multiply(q[:, i_q_h], k, dtype=qk_mul.dtype)
        qk_psum[:, i_q_h, :] = nisa.nc_matmul(
            qk_mul,
            ones,
            is_moving_onezero=True,
        )

    o_previous_scaled = nl.ndarray(
        (par_dim(TILE_SIZE), q_h_per_k_h, B_D_SIZE), dtype=o_buffer.dtype
    )
    max_ = qk_psum

    m_previous = nl.ndarray((par_dim(TILE_SIZE), q_h_per_k_h, 1), dtype=acc_type)
    m_previous[...] = nl.copy(m_buffer)
    m_buffer[...] = nl.maximum(m_previous, max_)

    alpha = nl.ndarray((par_dim(TILE_SIZE), q_h_per_k_h, 1), dtype=acc_type)
    bias = nl.zeros((par_dim(TILE_SIZE), 1), dtype=m_previous.dtype)
    alpha[...] = nisa.activation(
        np.exp,
        m_previous - m_buffer,
        scale=1.0,
        bias=bias,
    )
    o_previous_scaled[...] = nl.multiply(o_buffer, alpha)

    p_local = nl.ndarray((par_dim(TILE_SIZE), q_h_per_k_h, 1), dtype=kernel_dtype)
    ps = nl.ndarray(
        (par_dim(TILE_SIZE), q_h_per_k_h, 1),
        dtype=acc_type,
    )
    for i_q_h in nl.affine_range(q_h_per_k_h):
        p_local[:, i_q_h, :] = nisa.activation_reduce(
            np.exp,
            qk_psum[:, i_q_h, :],
            bias=-1 * m_buffer[:, i_q_h, :],
            scale=1.0,
            reduce_op=nl.add,
            reduce_res=ps[:, i_q_h, :],
            dtype=kernel_dtype,
        )

    pv_sbuf = nl.ndarray(
        (par_dim(TILE_SIZE), q_h_per_k_h, B_D_SIZE),
        dtype=np.float32,
        buffer=nl.sbuf,
    )
    pv_sbuf[...] = nisa.tensor_tensor(p_local, v, nl.multiply, dtype=np.float32)

    o_buffer[...] = nl.add(o_previous_scaled, pv_sbuf)

    l_prev = l_buffer * alpha
    l_buffer[...] = l_prev + ps


def prefill_prior_tokens(
    num_tiles_unrolled,
    query,
    key_cache,
    value_cache,
    m_buffer,
    l_buffer,
    o_buffer,
    tile_q_indices_sbuf,
    cur_masks,
    tile_masks,
    block_tables_sbuf,
    num_blocks_per_large_tile,
    block_size,
    kernel_dtype,
    acc_type,
    identity_for_transpose_k_hbm,
    identity_for_transpose_p_hbm,
    batch_id,
    head_id,
    q_h_per_k_h,
    softmax_scale,
    n_small_in_large_q_tile,
    INNER_Q_TILE_SIZE,
    LARGE_KV_TILE_SIZE,
    B_F_SIZE,
    B_D_SIZE,
):
    load_mask_locally = cur_masks is None
    identity_for_transpose_k = nl.load(identity_for_transpose_k_hbm)
    identity_for_transpose_p = nl.load(identity_for_transpose_p_hbm)
    # Max, LSE, Output, and Q buffers on sbuf, define them outside static loop for DMA skipping
    m_sbuf_tile = nl.zeros(
        (par_dim(INNER_Q_TILE_SIZE), n_small_in_large_q_tile, q_h_per_k_h, 1),
        dtype=acc_type,
        buffer=nl.sbuf,
    )
    l_sbuf_tile = nl.zeros(
        (par_dim(INNER_Q_TILE_SIZE), n_small_in_large_q_tile, q_h_per_k_h, 1),
        dtype=acc_type,
        buffer=nl.sbuf,
    )
    o_sbuf_tile = nl.zeros(
        (par_dim(INNER_Q_TILE_SIZE), n_small_in_large_q_tile, q_h_per_k_h, B_D_SIZE),
        dtype=acc_type,
        buffer=nl.sbuf,
    )
    num_loads = num_blocks_per_large_tile // B_P_SIZE
    k_load_buffer = nl.zeros(
        (par_dim(B_P_SIZE), num_loads, block_size * B_D_SIZE),
        dtype=key_cache.dtype,
    )
    v_load_buffer = nl.zeros(
        (par_dim(B_P_SIZE), num_loads * block_size * B_D_SIZE),
        dtype=value_cache.dtype,
    )

    for local_tile_idx in nl.sequential_range(num_tiles_unrolled):
        if load_mask_locally:
            cur_masks = load_indices(
                tile_masks[local_tile_idx],
                partition_size=INNER_Q_TILE_SIZE,
            )
        cur_k_tile, cur_v_tile = load_kv_tile_from_cache(
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=block_tables_sbuf,
            large_k_tile_idx=local_tile_idx,
            num_blocks_per_large_tile=num_blocks_per_large_tile,
            block_size=block_size,
            B_D_SIZE=B_D_SIZE,
            kernel_dtype=kernel_dtype,
            k_load_buffer=k_load_buffer,
            v_load_buffer=v_load_buffer,
            identity_for_transpose=identity_for_transpose_k,
        )
        # load aggregation buffer and q from HBM
        q_sbuf_tile_transposed = nl.ndarray(
            (
                par_dim(B_D_SIZE),
                q_h_per_k_h,
                n_small_in_large_q_tile,
                INNER_Q_TILE_SIZE,
            ),
            dtype=query.dtype,
            buffer=nl.sbuf,
        )
        # XXX: nl.zeros due to DMA skipping, otherwise, will get NaNs
        q_sbuf_tmp = nl.zeros(
            (
                par_dim(INNER_Q_TILE_SIZE),
                n_small_in_large_q_tile,
                q_h_per_k_h,
                B_D_SIZE,
            ),
            dtype=kernel_dtype,
            buffer=nl.sbuf,
        )
        for small_q_idx in nl.affine_range(n_small_in_large_q_tile):
            for i_q_h in nl.affine_range(q_h_per_k_h):
                i_p = nl.arange(INNER_Q_TILE_SIZE)[:, None]
                i_f = nl.arange(B_D_SIZE)[None, :]
                q_sbuf_tmp[i_p, small_q_idx, i_q_h, i_f] = nl.load(
                    query[
                        batch_id,
                        head_id * q_h_per_k_h + i_q_h,
                        tile_q_indices_sbuf[i_p, small_q_idx, local_tile_idx],
                        i_f,
                    ],
                    mode=oob_mode.skip,
                )
                q_tile_scaled = nl.multiply(
                    q_sbuf_tmp[:, small_q_idx, i_q_h, :],
                    softmax_scale,
                    dtype=kernel_dtype,
                )
                PF_transpose_with_PE(
                    q_tile_scaled,
                    q_sbuf_tile_transposed[:, i_q_h, small_q_idx, :],
                    identity_for_transpose=identity_for_transpose_p,
                    out_in_psum=False,
                )
        for small_q_idx in nl.affine_range(n_small_in_large_q_tile):
            i_p = nl.arange(INNER_Q_TILE_SIZE)[:, None, None]
            i_f_h = nl.arange(q_h_per_k_h)[None, :, None]
            i_f_d = nl.arange(1)[None, None, :]
            m_sbuf_tile[i_p, small_q_idx, i_f_h, i_f_d] = nl.load(
                m_buffer[
                    tile_q_indices_sbuf[i_p, small_q_idx, local_tile_idx], i_f_h, i_f_d
                ],
                mode=oob_mode.skip,
            )
            l_sbuf_tile[i_p, small_q_idx, i_f_h, i_f_d] = nl.load(
                l_buffer[
                    tile_q_indices_sbuf[i_p, small_q_idx, local_tile_idx], i_f_h, i_f_d
                ],
                mode=oob_mode.skip,
            )
            i_p = nl.arange(INNER_Q_TILE_SIZE)[:, None, None]
            i_f_h = nl.arange(q_h_per_k_h)[None, :, None]
            i_f_d = nl.arange(B_D_SIZE)[None, None, :]
            o_sbuf_tile[i_p, small_q_idx, i_f_h, i_f_d] = nl.load(
                o_buffer[
                    tile_q_indices_sbuf[i_p, small_q_idx, local_tile_idx], i_f_h, i_f_d
                ],
                mode=oob_mode.skip,
            )

        for small_q_idx in nl.affine_range(n_small_in_large_q_tile):
            for i_q_h in nl.affine_range(q_h_per_k_h):
                q_tile = q_sbuf_tile_transposed[:, i_q_h, small_q_idx]
                if load_mask_locally:
                    # for static loop
                    cur_mask_tile = cur_masks[:, small_q_idx, :]
                else:
                    # for dynamic while loop (with static unrolling)
                    cur_mask_tile = cur_masks[
                        :, local_tile_idx * n_small_in_large_q_tile + small_q_idx
                    ]
                _flash_attention_core(
                    q_local_tile=q_tile,
                    k=cur_k_tile,
                    v=cur_v_tile,
                    o_buffer=o_sbuf_tile[:, small_q_idx, i_q_h],
                    l_buffer=l_sbuf_tile[:, small_q_idx, i_q_h],
                    m_buffer=m_sbuf_tile[:, small_q_idx, i_q_h],
                    kernel_dtype=kernel_dtype,
                    acc_type=acc_type,
                    tile_mask=cur_mask_tile,
                    identity_for_transpose=identity_for_transpose_p,
                    use_causal_mask=False,
                    q_tile_idx=None,
                    Q_TILE_SIZE=INNER_Q_TILE_SIZE,
                    LARGE_KV_TILE_SIZE=LARGE_KV_TILE_SIZE,
                    B_F_SIZE=B_F_SIZE,
                    B_D_SIZE=B_D_SIZE,
                )

        for small_q_idx in nl.affine_range(n_small_in_large_q_tile):
            i_p = nl.arange(INNER_Q_TILE_SIZE)[:, None, None]
            i_f_h = nl.arange(q_h_per_k_h)[None, :, None]
            i_f_d = nl.arange(B_D_SIZE)[None, None, :]
            nl.store(
                o_buffer[
                    tile_q_indices_sbuf[i_p, small_q_idx, local_tile_idx], i_f_h, i_f_d
                ],
                o_sbuf_tile[i_p, small_q_idx, i_f_h, i_f_d],
                mode=oob_mode.skip,
            )
            i_p = nl.arange(INNER_Q_TILE_SIZE)[:, None, None]
            i_f_h = nl.arange(q_h_per_k_h)[None, :, None]
            i_f_d = nl.arange(1)[None, None, :]
            nl.store(
                m_buffer[
                    tile_q_indices_sbuf[i_p, small_q_idx, local_tile_idx], i_f_h, i_f_d
                ],
                m_sbuf_tile[i_p, small_q_idx, i_f_h, i_f_d],
                mode=oob_mode.skip,
            )
            nl.store(
                l_buffer[
                    tile_q_indices_sbuf[i_p, small_q_idx, local_tile_idx], i_f_h, i_f_d
                ],
                l_sbuf_tile[i_p, small_q_idx, i_f_h, i_f_d],
                mode=oob_mode.skip,
            )


def prefill_active_tokens_and_epilogue(
    o,
    query,
    key,
    value,
    active_mask,
    softmax_scale,
    m_buffer,
    l_buffer,
    o_buffer,
    ACTIVE_Q_TILE_SIZE,
    seqlen_q,
    batch_id,
    head_id,
    q_h_per_k_h,
    kernel_dtype,
    acc_type,
    B_F_SIZE,
    B_D_SIZE,
):
    # -------- Load l, m, o back to SBUF from HBM ------------ #
    num_active_tiles = seqlen_q // ACTIVE_Q_TILE_SIZE
    assert seqlen_q % ACTIVE_Q_TILE_SIZE == 0

    o_buffer_sbuf = nl.ndarray(
        (par_dim(ACTIVE_Q_TILE_SIZE), num_active_tiles, q_h_per_k_h, B_D_SIZE),
        dtype=acc_type,
    )
    m_buffer_sbuf = nl.ndarray(
        (par_dim(ACTIVE_Q_TILE_SIZE), num_active_tiles, q_h_per_k_h, 1),
        dtype=acc_type,
    )
    l_buffer_sbuf = nl.ndarray(
        (par_dim(ACTIVE_Q_TILE_SIZE), num_active_tiles, q_h_per_k_h, 1),
        dtype=acc_type,
    )

    for i in nl.affine_range(num_active_tiles):
        o_buffer_sbuf[:, i] = nl.load(
            o_buffer[nl.ds(i * ACTIVE_Q_TILE_SIZE, ACTIVE_Q_TILE_SIZE)]
        )
        l_buffer_sbuf[:, i] = nl.load(
            l_buffer[nl.ds(i * ACTIVE_Q_TILE_SIZE, ACTIVE_Q_TILE_SIZE)]
        )
        m_buffer_sbuf[:, i] = nl.load(
            m_buffer[nl.ds(i * ACTIVE_Q_TILE_SIZE, ACTIVE_Q_TILE_SIZE)]
        )

    # compute attention between input query, key and value
    if key is not None and value is not None:
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

        identity_for_transpose_p = create_identity_for_transpose(
            cur_v_tile, ACTIVE_Q_TILE_SIZE
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
                    (ACTIVE_Q_TILE_SIZE, B_D_SIZE), dtype=query.dtype
                )
                q_sbuf_tile[:, :] = nl.load(
                    q_hbm_tile[nl.ds(i * ACTIVE_Q_TILE_SIZE, ACTIVE_Q_TILE_SIZE), :],
                )  # load (d, 128) tile in SBUF
                q_tile_scaled = nl.ndarray(
                    (ACTIVE_Q_TILE_SIZE, B_D_SIZE), dtype=kernel_dtype
                )
                q_tile_scaled[:, :] = nl.multiply(
                    q_sbuf_tile, softmax_scale, dtype=kernel_dtype
                )
                q_tile = nl.ndarray((B_D_SIZE, ACTIVE_Q_TILE_SIZE), dtype=kernel_dtype)
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
                    o_buffer=o_buffer_sbuf[:, i, i_q_h],
                    l_buffer=l_buffer_sbuf[:, i, i_q_h],
                    m_buffer=m_buffer_sbuf[:, i, i_q_h],
                    kernel_dtype=kernel_dtype,
                    acc_type=acc_type,
                    tile_mask=cur_mask,
                    identity_for_transpose=identity_for_transpose_p,
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
            (par_dim(ACTIVE_Q_TILE_SIZE), q_h_per_k_h, B_D_SIZE), dtype=kernel_dtype
        )
        out[...] = nl.multiply(
            o_buffer_sbuf[:, i],
            1.0
            / l_buffer_sbuf[
                :, i
            ],  # XXX: l is 0 in padded tokens, warning in simulation
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


def decode_prior_tokens(
    num_tiles_unrolled,
    query_sbuf,
    key_cache,
    value_cache,
    lmo_buffer,
    m_next_tile,
    l_next_tile,
    o_next_tile,
    q_offsets,
    block_tables_sbuf,
    tile_mask_sbuf,
    q_update_pred_sbuf,
    q_update_pred_broadcast,
    num_blocks_per_large_tile,
    block_size,
    kernel_dtype,
    acc_type,
    q_h_per_k_h,
    B_D_SIZE,
    MULTI_BUFFER_SIZE,
    identity_for_transpose_m_step1_hbm,
    identity_for_transpose_m_step2_hbm,
    identity_for_transpose_k_hbm,
):

    identity_for_transpose_k = nl.load(identity_for_transpose_k_hbm)
    identity_for_transpose_m_step1 = nl.load(identity_for_transpose_m_step1_hbm)
    identity_for_transpose_m_step2 = nl.load(identity_for_transpose_m_step2_hbm)

    # XXX: no dma skipping for decode kernel, no need to zero
    num_loads = num_blocks_per_large_tile // B_P_SIZE
    k_load_buffer = nl.ndarray(
        (par_dim(B_D_SIZE), MULTI_BUFFER_SIZE, num_loads * block_size * B_P_SIZE),
        dtype=kernel_dtype,
    )
    v_load_buffer = nl.ndarray(
        (par_dim(B_P_SIZE), MULTI_BUFFER_SIZE, num_loads * block_size * B_D_SIZE),
        dtype=kernel_dtype,
    )
    # load first iteration of key cache
    k_load_buffer_reshaped = k_load_buffer.reshape(
        (B_D_SIZE, MULTI_BUFFER_SIZE, num_loads, block_size, B_P_SIZE),
    )
    for load_idx in nl.affine_range(num_loads):
        i_p = nl.arange(B_P_SIZE)[:, None]
        i_f = nl.arange(block_size * B_D_SIZE)[None, :]
        loaded = nl.ndarray((B_P_SIZE, block_size * B_D_SIZE), dtype=key_cache.dtype)
        loaded[i_p, i_f] = nl.load(key_cache[block_tables_sbuf[i_p, load_idx, 0], i_f])
        for tb_i in nl.affine_range(block_size):
            if loaded.dtype != kernel_dtype:
                k_src = nl.copy(
                    loaded[:, nl.ds(tb_i * B_D_SIZE, B_D_SIZE)],
                    dtype=kernel_dtype,
                )
            else:
                k_src = loaded[:, nl.ds(tb_i * B_D_SIZE, B_D_SIZE)]
            PF_transpose_with_PE(
                src=k_src,
                out=k_load_buffer_reshaped[:, 0, load_idx, tb_i],
                identity_for_transpose=identity_for_transpose_k,
            )

    # load value cache
    for load_idx in nl.affine_range(num_loads):
        i_p = nl.arange(B_P_SIZE)[:, None]
        i_f = nl.arange(block_size * B_D_SIZE)[None, :]
        if kernel_dtype == value_cache.dtype:
            v_load_buffer[i_p, 0, i_f + load_idx * block_size * B_D_SIZE] = nl.load(
                value_cache[block_tables_sbuf[i_p, load_idx, 0], i_f],
            )
        else:
            loaded = nl.ndarray(
                (B_P_SIZE, block_size * B_D_SIZE), dtype=value_cache.dtype
            )
            loaded[...] = nl.load(
                value_cache[block_tables_sbuf[i_p, load_idx, 0], i_f],
            )
            v_load_buffer[i_p, 0, i_f + load_idx * block_size * B_D_SIZE] = nl.copy(
                loaded,
                dtype=kernel_dtype,
            )

    NEG_INF = -9984.0  # Magic number to replace -inf similar to what Tensorizer uses
    num_tiles_padded = num_tiles_unrolled + 1
    if num_tiles_unrolled > B_P_SIZE:
        # pad to multiple of 128 for PE transpose during write out
        num_tiles_padded = pad_to_multiple(num_tiles_padded, B_P_SIZE)
    o_buffer_sbuf = nl.zeros(
        (par_dim(B_D_SIZE), num_tiles_padded, q_h_per_k_h),
        dtype=acc_type,
    )
    m_buffer_sbuf = nl.full(
        (par_dim(1), num_tiles_padded, q_h_per_k_h),
        NEG_INF,
        dtype=acc_type,
    )
    l_buffer_sbuf = nl.zeros(
        (par_dim(1), num_tiles_padded, q_h_per_k_h),
        dtype=acc_type,
    )
    if q_offsets is not None:
        # dynamic loop has previous states from last dynamic loop step
        o_buffer_sbuf[:, 0, :] = nl.load(o_next_tile)
        l_buffer_sbuf[:, 0, :] = nl.load(l_next_tile)
        m_buffer_sbuf[:, 0, :] = nl.load(m_next_tile)

    kq_res_ones = nl.ones(
        (par_dim(B_P_SIZE), tile_mask_sbuf.shape[-1], q_h_per_k_h),
        dtype=acc_type,
    )
    sumexp_ones = nl.ones((par_dim(B_P_SIZE), 1), dtype=kernel_dtype)

    for local_tile_idx in nl.sequential_range(num_tiles_unrolled):
        _flash_attention_core_kq_matmul(
            q_local_tile=query_sbuf,
            o_buffer_sbuf=o_buffer_sbuf,
            l_buffer_sbuf=l_buffer_sbuf,
            m_buffer_sbuf=m_buffer_sbuf,
            kernel_dtype=kernel_dtype,
            acc_type=acc_type,
            q_update_pred_sbuf=q_update_pred_sbuf,
            q_update_pred_broadcast=q_update_pred_broadcast,
            tile_mask=tile_mask_sbuf,
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables_sbuf=block_tables_sbuf,
            large_tile_idx=local_tile_idx,
            MULTI_BUFFER_SIZE=MULTI_BUFFER_SIZE,
            num_blocks_per_large_tile=num_blocks_per_large_tile,
            block_size=block_size,
            k_load_buffer=k_load_buffer,
            v_load_buffer=v_load_buffer,
            kq_res_ones=kq_res_ones,
            sumexp_ones=sumexp_ones,
            identity_for_transpose_k=identity_for_transpose_k,
            identity_for_transpose_m_step1=identity_for_transpose_m_step1,
            identity_for_transpose_m_step2=identity_for_transpose_m_step2,
        )

    if q_offsets is not None:
        nl.store(o_next_tile, o_buffer_sbuf[:, num_tiles_unrolled, :])
        nl.store(l_next_tile, l_buffer_sbuf[:, num_tiles_unrolled, :])
        nl.store(m_next_tile, m_buffer_sbuf[:, num_tiles_unrolled, :])
    # store L, M, O in (seqlen, h, d) format
    write_tile_size = min(num_tiles_unrolled, B_P_SIZE)
    identity_for_transpose_o = create_identity_for_transpose(o_buffer_sbuf, B_D_SIZE)
    identity_for_transpose_lm = nl.ones((1, 1), dtype=acc_type)
    for i in nl.affine_range(ceil_div(num_tiles_unrolled, write_tile_size)):
        lmo_sbuf = nl.ndarray(
            (write_tile_size, q_h_per_k_h, B_D_SIZE + 2),
            dtype=o_buffer_sbuf.dtype,
        )
        for i_q_h in nl.affine_range(q_h_per_k_h):
            o_tmp = nl.ndarray(
                (write_tile_size, B_D_SIZE), dtype=acc_type, buffer=nl.psum
            )
            PF_transpose_with_PE(
                src=o_buffer_sbuf[
                    :,
                    nl.ds(i * write_tile_size, write_tile_size),
                    i_q_h,
                ],
                out=o_tmp,
                identity_for_transpose=identity_for_transpose_o,
                out_in_psum=True,
            )
            lmo_sbuf[:, i_q_h, nl.ds(0, B_D_SIZE)] = nl.copy(o_tmp)
            l_tmp = nl.ndarray((write_tile_size, 1), dtype=acc_type, buffer=nl.psum)
            PF_transpose_with_PE(
                src=l_buffer_sbuf[
                    :,
                    nl.ds(i * write_tile_size, write_tile_size),
                    i_q_h,
                ],
                out=l_tmp,
                identity_for_transpose=identity_for_transpose_lm,
                out_in_psum=True,
            )
            lmo_sbuf[:, i_q_h, nl.ds(B_D_SIZE, 1)] = nl.copy(l_tmp)
            m_tmp = nl.ndarray((write_tile_size, 1), dtype=acc_type, buffer=nl.psum)
            PF_transpose_with_PE(
                src=m_buffer_sbuf[
                    :,
                    nl.ds(i * write_tile_size, write_tile_size),
                    i_q_h,
                ],
                out=m_tmp,
                identity_for_transpose=identity_for_transpose_lm,
                out_in_psum=True,
            )
            lmo_sbuf[:, i_q_h, nl.ds(B_D_SIZE + 1, 1)] = nl.copy(m_tmp)
        i_q = nl.arange(write_tile_size)[:, None, None]
        i_h = nl.arange(q_h_per_k_h)[None, :, None]
        i_d = nl.arange(B_D_SIZE + 2)[None, None, :]
        if q_offsets is not None:
            nl.store(
                lmo_buffer[q_offsets[i_q + i * write_tile_size, 0], i_h, i_d],
                lmo_sbuf[i_q, i_h, i_d],
                mask=(i_q + i * write_tile_size < q_offsets.shape[0]),
            )
        else:
            nl.store(
                lmo_buffer[i_q + i * write_tile_size, i_h, i_d],
                lmo_sbuf[i_q, i_h, i_d],
                mask=(i_q + i * write_tile_size < num_tiles_unrolled),
            )


def decode_active_tokens_and_epilogue(
    o,
    query,
    key,
    value,
    lmo_buffer,
    softmax_scale,
    last_tile_indices_sbuf,
    seqlen_q,
    batch_id,
    head_id,
    q_h_per_k_h,
    kernel_dtype,
    acc_type,
    B_D_SIZE,
):
    ACTIVE_Q_TILE_SIZE = min(seqlen_q, B_P_SIZE)
    num_active_tiles = seqlen_q // ACTIVE_Q_TILE_SIZE
    assert seqlen_q % ACTIVE_Q_TILE_SIZE == 0
    assert last_tile_indices_sbuf.shape == (ACTIVE_Q_TILE_SIZE, num_active_tiles)

    o_buffer_sbuf = nl.ndarray(
        (par_dim(ACTIVE_Q_TILE_SIZE), num_active_tiles, q_h_per_k_h, B_D_SIZE),
        dtype=acc_type,
    )
    m_buffer_sbuf = nl.ndarray(
        (par_dim(ACTIVE_Q_TILE_SIZE), num_active_tiles, q_h_per_k_h, 1),
        dtype=acc_type,
    )
    l_buffer_sbuf = nl.ndarray(
        (par_dim(ACTIVE_Q_TILE_SIZE), num_active_tiles, q_h_per_k_h, 1),
        dtype=acc_type,
    )
    for i in nl.affine_range(num_active_tiles):
        lmo_tmp = nl.ndarray(
            (par_dim(ACTIVE_Q_TILE_SIZE), q_h_per_k_h, B_D_SIZE + 2),
            dtype=acc_type,
        )
        i_q = nl.arange(ACTIVE_Q_TILE_SIZE)[:, None, None]
        i_h = nl.arange(q_h_per_k_h)[None, :, None]
        i_d = nl.arange(B_D_SIZE + 2)[None, None, :]
        lmo_tmp[i_q, i_h, i_d] = nl.load(
            lmo_buffer[last_tile_indices_sbuf[i_q, i], i_h, i_d]
        )
        o_buffer_sbuf[:, i, :, :] = nl.copy(lmo_tmp[:, :, nl.ds(0, B_D_SIZE)])
        l_buffer_sbuf[:, i, :, :] = nl.copy(lmo_tmp[:, :, nl.ds(B_D_SIZE, 1)])
        m_buffer_sbuf[:, i, :, :] = nl.copy(lmo_tmp[:, :, nl.ds(B_D_SIZE + 1, 1)])

    # compute attention between input query, key and value
    if key is not None and value is not None:
        cur_q_tile = nl.ndarray(
            (par_dim(B_D_SIZE), num_active_tiles, q_h_per_k_h, ACTIVE_Q_TILE_SIZE),
            dtype=kernel_dtype,
        )
        identity_for_transpose_q = create_identity_for_transpose(
            cur_q_tile, ACTIVE_Q_TILE_SIZE
        )
        for i in nl.affine_range(num_active_tiles):
            for i_q_h in nl.affine_range(q_h_per_k_h):
                q_hbm_tile = query[
                    batch_id,
                    head_id * q_h_per_k_h + i_q_h,
                    nl.ds(i * ACTIVE_Q_TILE_SIZE, ACTIVE_Q_TILE_SIZE),
                    :,
                ]
                q_sbuf_tile = nl.load(q_hbm_tile)
                q_tile_scaled = nl.multiply(
                    q_sbuf_tile, softmax_scale, dtype=kernel_dtype
                )
                PF_transpose_with_PE(
                    q_tile_scaled,
                    cur_q_tile[:, i, i_q_h],
                    identity_for_transpose=identity_for_transpose_q,
                    out_in_psum=False,
                )
        cur_k_tile = nl.ndarray(
            (par_dim(B_D_SIZE), num_active_tiles, ACTIVE_Q_TILE_SIZE),
            dtype=kernel_dtype,
        )
        cur_v_tile = nl.ndarray(
            (par_dim(ACTIVE_Q_TILE_SIZE), num_active_tiles, 1, B_D_SIZE),
            dtype=kernel_dtype,
        )
        for i in nl.affine_range(num_active_tiles):
            cur_k_tile[:, i, :] = nl.load(
                key[
                    batch_id,
                    head_id,
                    :,
                    nl.ds(i * ACTIVE_Q_TILE_SIZE, ACTIVE_Q_TILE_SIZE),
                ],
                dtype=kernel_dtype,
            )
            cur_v_tile[:, i, 0, :] = nl.load(
                value[
                    batch_id,
                    head_id,
                    nl.ds(i * ACTIVE_Q_TILE_SIZE, ACTIVE_Q_TILE_SIZE),
                    :,
                ],
                dtype=kernel_dtype,
            )

        for i in nl.affine_range(num_active_tiles):
            _active_attention_core_batched(
                q=cur_q_tile[:, i],
                k=cur_k_tile[:, i],
                v=cur_v_tile[:, i],
                o_buffer=o_buffer_sbuf[:, i],
                l_buffer=l_buffer_sbuf[:, i],
                m_buffer=m_buffer_sbuf[:, i],
                q_h_per_k_h=q_h_per_k_h,
                kernel_dtype=kernel_dtype,
                acc_type=acc_type,
                TILE_SIZE=ACTIVE_Q_TILE_SIZE,
                B_D_SIZE=B_D_SIZE,
            )

    # -------- write output to buffer on HBM ------------ #
    for i in nl.affine_range(num_active_tiles):
        out = nl.ndarray(
            (par_dim(ACTIVE_Q_TILE_SIZE), q_h_per_k_h, B_D_SIZE), dtype=kernel_dtype
        )
        out[...] = nl.multiply(
            o_buffer_sbuf[:, i],
            1.0
            / l_buffer_sbuf[
                :, i
            ],  # XXX: l is 0 in padded tokens, warning in simulation
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
